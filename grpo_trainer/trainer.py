from typing import List, Dict, Optional, Tuple
import torch
import torch.optim as optim
from torch.nn import functional as F
import gc
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import os
from typing import Dict

from .config import GRPOConfig
from .reward import RewardCalculator
from .conversation import ConversationGenerator, ConversationTurn
from huggingface_hub import HfApi, login


class GRPOTrainer:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: GRPOConfig,
        medical_data,
        patient_model: Optional[AutoModelForCausalLM] = None,
        doctor_model: Optional[AutoModelForCausalLM] = None
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.reward_calculator = RewardCalculator()

        self.patient_model = patient_model or self._init_model(eval_mode=True)
        self.doctor_model = doctor_model or self._init_model(eval_mode=False)

        self.optimizer = optim.AdamW(
            self.doctor_model.parameters(),
            lr=self.config.learning_rate
        )
        self.conversation_generator = ConversationGenerator(
            self.config,
            self.tokenizer,
            self.patient_model,
            self.doctor_model,
            medical_data
        )

        self.csv_path = "./branch1_batch4_lr2e-7_kl0_w_pt_stipulation_training_log.csv"
        self._init_csv_logger()

    def _init_csv_logger(self):
        core_cols = [
            "total_loss", "policy_loss", "kl_loss", "mean_reward",
            "degenerate_count", "final_branch_count"
        ]
    
        n = self.config.runs_per_case          
        for i in range(1, n + 1):
            core_cols += [f"conversation_{i}",
                          f"diagnosis_score_{i}",
                          f"llm_diagnosis_{i}"]
    
        # (re)-create file if header missing / wrong
        if not os.path.exists(self.csv_path):
            recreate = True
        else:
            with open(self.csv_path) as f:
                recreate = f.readline().strip().split(",") != core_cols
    
        if recreate:
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(core_cols)

    def _init_model(self, eval_mode: bool) -> AutoModelForCausalLM:
        print(f"Initializing {'patient' if eval_mode else 'doctor'} model")

        if not eval_mode:
            print(f"========================= 100% ==========================")
        dtype = torch.float16 if self.config.fp16 else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map="auto" if self.config.device == "cuda" else None,
            trust_remote_code=True
        )

        if eval_mode:
            model.eval()
        else:
            if self.config.gradient_checkpointing:
                model.gradient_checkpointing_enable()

        return model

    def _is_valid_doctor_response(self, response: str) -> bool:
        word_count = len(response.strip().split())
        has_question = '?' in response
        contains_patient_label = "Patient" in response  # case-sensitive
        return word_count >= 4 and has_question and not contains_patient_label


    def train_step(
        self,
        queries: torch.Tensor,
        responses: torch.Tensor,
        rewards: torch.Tensor,
        raw_rewards: torch.Tensor,
        diagnosis_scores: torch.Tensor
    ) -> Dict[str, float]:
        self.doctor_model.train()

        outputs = self.doctor_model(input_ids=queries, labels=responses)
        logprobs = F.log_softmax(outputs.logits, dim=-1)
        response_logprobs = torch.gather(logprobs, -1, responses.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            ref_outputs = self.patient_model(input_ids=queries, labels=responses)
            ref_logprobs = F.log_softmax(ref_outputs.logits, dim=-1)
            old_logprobs = torch.gather(ref_logprobs, -1, responses.unsqueeze(-1)).squeeze(-1)

        advantages = rewards.unsqueeze(-1).expand(-1, responses.shape[1])   
        ratios = torch.exp(torch.clamp(response_logprobs - old_logprobs, -6, 6)) 
        policy_loss = -(ratios * advantages).mean()                              
        kl_loss = self.config.kl_coeff * (old_logprobs - response_logprobs).mean()   
        total_loss = policy_loss + kl_loss            

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.doctor_model.parameters(), 0.5)
        self.optimizer.step()

        print("📈 Diagnosis scores used in reward calculation:")
        for i, score in enumerate(diagnosis_scores.tolist()):
            print(f"  Sample {i+1}: {score:.4f}")

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            'mean_reward': diagnosis_scores.mean().item()
        }

    def train_on_conversation_tree(self, patient_input: str) -> Dict[str, float]:
        print(f"\n=== Starting GRPO Training ===")
        print(f"Branches per turn: {self.config.branches}")
        print(f"Max turns: {self.config.num_turns}")
        print(f"Initial input: {patient_input[:50]}...")
    
        try:
            self.conversation_generator.degenerate_count = 0
            branches = self.conversation_generator.generate_branched_conversation(patient_input)
            final_branch_count = len(branches)
            training_data = self._prepare_training_data(branches)
    
            if training_data is None:
                print("No valid training data generated")
                return {}
    
            queries, responses, norm_rewards, raw_rewards, diagnosis_scores = training_data
            print("All raw_rewards:", raw_r.cpu().tolist())
            

            print(f"📊 Number of training samples this step: {len(queries)}")

            decoded_queries = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
            decoded_responses = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
    
            print("\n=== Training Input Samples ===")
            for i in range(len(decoded_queries)):
                print(f"\n🗂 Sample {i}")
                print(f"📋 Query:\n{decoded_queries[i]}")
                print(f"🤖 Response:\n{decoded_responses[i]}")
                print(f"🏅 Normalized Reward: {norm_rewards[i].item():.4f}")
                print("-" * 50)
    
            stats = self.train_step(queries, responses, norm_rewards, raw_rewards, diagnosis_scores)
            self._print_training_stats(stats, branches)
    
            # Log all results as a single row per conversation
            self._log_conversation_stats(
                stats,
                diagnosis_scores,
                decoded_queries,
                decoded_responses,
                llm_diagnoses,  # ← NEW ARG
                branches,
                self.conversation_generator.degenerate_count,
                final_branch_count
            )

            return stats
    
        except Exception as e:
            print(f"\n❌ Training Error: {str(e)}")
            raise
        finally:
            self._cleanup()
    

            
    def _get_full_conversation_text(self, branch: List[ConversationTurn]) -> str:
        return "\n".join(
            f"{turn.speaker.capitalize()}: {turn.text.strip()}"
            for turn in branch
        )
    def _split_query_response(                   # <--  ADD THIS METHOD
            self,
            branch: List[ConversationTurn]
    ) -> Tuple[str, str]:
        """
        Return:
            query   – full conversation *up to* the last **patient** turn
            response – the *next* doctor turn that should be learned
        """
        # traverse from the end until we find the last doctor turn
        for idx in reversed(range(len(branch))):
            if branch[idx].speaker.lower() == "doctor":
                doctor_turn = branch[idx]
                patient_context = branch[:idx]          # everything before that doctor
                break
        else:
            raise ValueError("Branch has no doctor turn")
    
        query = self._get_full_conversation_text(patient_context)
        response = doctor_turn.text
        return query, response
    
    
    # ------------------------------------------------------------
    # Main routine
    # ------------------------------------------------------------
    def _prepare_training_data(self, branches: List[List[ConversationTurn]]):
        """
        Prepares training data:
        - Leaf diagnosis scores and rewards computed normally.
        - If a leaf doctor response is invalid → diag_score = 0.0
        - Parent turns inherit child score (even if parent turn is valid/invalid).
        """
        all_queries, all_responses, all_rewards = [], [], []
        all_diag_scores, all_llm_diagnoses = [], []
    
        leaves = [b for b in branches if len(b) == self.config.num_turns]
        if not leaves:
            return None
    
        diagnoses = []
        diag_scores = []
    
        # Step 1: Evaluate each leaf
        for br in leaves:
            full_txt = self._get_full_conversation_text(br)
            diagnosis = self.reward_calculator.get_llm_diagnosis(full_txt)
    
            # Get doctor response (last doctor turn)
            _, doc_response = self._split_query_response(br)
            if not self._is_valid_doctor_response(doc_response):
                print(f"⚠️ Invalid leaf doctor response: \"{doc_response}\"")
                score = 0.0
                diagnosis = "skipped"
            else:
                score = self.reward_calculator.compare_diagnoses(diagnosis, br[0].text)
    
            diagnoses.append(diagnosis)
            diag_scores.append(score)
    
        # Step 2: Compute rewards and save training data for leaves
        for i, (br, score) in enumerate(zip(leaves, diag_scores)):
            others = diag_scores[:i] + diag_scores[i+1:]
            mean_others = np.mean(others) if others else 0
            rel_reward = 0.5 + 0.5 * (score - mean_others)
    
            qry, rsp = self._split_query_response(br)
            all_queries.append(qry)
            all_responses.append(rsp)
            all_rewards.append(rel_reward)
            all_diag_scores.append(score)
            all_llm_diagnoses.append(diagnoses[i])
    
        # Step 3: Add parents (inherit diagnosis/score/reward from leaf)
        for i, br in enumerate(leaves):
            parent_branch = br[:-2]
            qry, rsp = self._split_query_response(parent_branch)
    
            parent_diag = diag_scores[i]         # always inherit from leaf
            parent_diag_str = diagnoses[i]
    
            if not self._is_valid_doctor_response(rsp):
                print(f"⚠️ Invalid parent doctor response: \"{rsp}\"")
                parent_diag = 0.0
                parent_diag_str = "skipped"
    
            all_queries.append(qry)
            all_responses.append(rsp)
            all_rewards.append(all_rewards[i])         # same reward as leaf
            all_diag_scores.append(parent_diag)
            all_llm_diagnoses.append(parent_diag_str)
    
        if not all_queries:
            return None
    
        # Tokenize
        tok_args = dict(padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        queries_t = self.tokenizer(all_queries, **tok_args)["input_ids"].to(self.config.device)
        responses_t = self.tokenizer(all_responses, **tok_args)["input_ids"].to(self.config.device)
    
        # Normalize rewards
        raw_r = torch.tensor(all_rewards, device=self.config.device).float()
        if raw_r.std().item() < 1e-6:
            print("⚠️ Skipping batch with near-zero reward variance")
            return None
    
        norm_r = (raw_r - raw_r.mean()) / (raw_r.std() + 1e-8)
        diag_t = torch.tensor(all_diag_scores, device=self.config.device).float()
    
        # FULL LOG
        print("\n📊 === Prepared Training Samples ===")
        for i in range(len(all_queries)):
            print(f"\n🧾 Sample {i+1}")
            print(f"📋 Query:\n{all_queries[i]}")
            print(f"🤖 Response:\n{all_responses[i]}")
            print(f"🧠 Diagnosis: {all_llm_diagnoses[i]}")
            print(f"📈 Diagnosis Score: {all_diag_scores[i]:.4f}")
            print(f"🏅 Raw Reward: {all_rewards[i]:.4f}")
            print(f"📊 Normalized Reward: {norm_r[i].item():+.4f}")
            print("-" * 60)
    
        return queries_t, responses_t, norm_r, raw_r, diag_t, all_llm_diagnoses


    def _split_query_response(self, branch: List[ConversationTurn]) -> Tuple[str, str]:
        for idx in reversed(range(len(branch))):
            if branch[idx].speaker.lower() == "doctor":
                doctor_turn = branch[idx]
                patient_context = branch[:idx]
                break
        else:
            raise ValueError("Branch has no doctor turn")

        query = self._get_full_conversation_text(patient_context)
        response = doctor_turn.text
        return query, response


    def _log_conversation_stats(
        self,
        stats: Dict[str, float],
        diagnosis_scores,
        decoded_queries,
        decoded_responses,
        llm_diagnoses,
        branches,
        degenerate_count,
        final_branch_count
    ):
        """
        Append **one** CSV row for this case:
          • core losses / reward
          • counts
          • up to 16 triples [conversation_i , diagnosis_score_i , llm_diagnosis_i]
        """
    
        row = [
            stats['total_loss'],
            stats['policy_loss'],
            stats['kl_loss'],
            stats['mean_reward'],
            degenerate_count,
            final_branch_count
        ]
    
        # --- gather up-to-16 final doctor-patient exchanges ---
        for i in range(4):
            if i < len(decoded_queries):
                convo = f"Patient: {decoded_queries[i].strip()}\nDoctor: {decoded_responses[i].strip()}"
                diag_score = diagnosis_scores[i].item()
                llm_diag   = llm_diagnoses[i]  # ✅ use directly from reward step
            else:
                convo, diag_score, llm_diag = "", 0.0, ""
            row.extend([convo, diag_score, llm_diag])
    
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)


    def train_on_case(self, patient_input: str) -> Dict[str, float]:
            """
            Run `config.runs_per_case` separate 1-branch conversations for the
            same patient case, then do ONE optimizer step on the union.
            """
            all_branches: List[List[ConversationTurn]] = []
            self.conversation_generator.degenerate_count = 0  # reset counter
    
            for run_id in range(self.config.runs_per_case):
                print(f"🔁  Conversation run {run_id+1}/{self.config.runs_per_case}")
                branches = self.conversation_generator.generate_branched_conversation(
                    patient_input
                )
                all_branches.extend(branches)           # branches == 1 element
    
            # keep track of how many final branches we actually have
            final_branch_count = len(all_branches)
    
            # ==== the rest is exactly what train_on_conversation_tree did ====
            training_data = self._prepare_training_data(all_branches)
            if training_data is None:
                print("No valid training data generated")
                return {}
    
            queries, responses, norm_rewards, raw_rewards, diagnosis_scores, llm_diagnoses = training_data
    
            decoded_queries   = self.tokenizer.batch_decode(queries,   skip_special_tokens=True)
            decoded_responses = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
    
            stats = self.train_step(
                queries, responses, norm_rewards, raw_rewards, diagnosis_scores
            )
            self._print_training_stats(stats, all_branches)
    
            # one log-row per CASE
            self._log_conversation_stats(
                stats,
                diagnosis_scores,
                decoded_queries,
                decoded_responses,
                llm_diagnoses,
                all_branches,
                self.conversation_generator.degenerate_count,
                final_branch_count
            )
            return stats
    


    def _print_training_stats(self, stats: Dict[str, float], branches: List[List[ConversationTurn]]):
        print("\n=== Training Results ===")
        print(f"Total Loss: {stats['total_loss']:.4f}")
        print(f"Policy Loss: {stats['policy_loss']:.4f}")
        print(f"KL Loss: {stats['kl_loss']:.4f}")
        print(f"Avg Reward: {stats['mean_reward']:.2f}")
        print(f"Processed {len(branches)} branches")

    def _cleanup(self):
        torch.cuda.empty_cache()
        gc.collect()

    def save_model_locally(self, save_path: str = "./saved_model"):
        print(f"💾 Saving model locally to {save_path}")
        if os.path.exists(save_path):
            for f in os.listdir(save_path):
                os.remove(os.path.join(save_path, f))
        else:
            os.makedirs(save_path, exist_ok=True)

        self.doctor_model.save_pretrained(save_path, safe_serialization=False)
        self.tokenizer.save_pretrained(save_path)


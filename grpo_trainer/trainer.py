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

        self.csv_path = "./branch4_lr2e-7_training_log_ref-old_kl0_log_w_patient_stipulation.csv"
        self._init_csv_logger()
        self.training_step_count = 0

    def _init_csv_logger(self):
        """
        Ensure the CSV has the correct header; if not, recreate it.
        """
        header = [
            "total_loss", "policy_loss", "kl_loss", "mean_reward",
            "degenerate_count", "final_branch_count"
        ]
        for i in range(1, 17):
            header += [f"conversation_{i}", f"diagnosis_score_{i}", f"llm_diagnosis_{i}"]
    
        # Create or replace file if header is wrong / missing
        if not os.path.exists(self.csv_path):
            recreate = True
        else:
            with open(self.csv_path, newline="") as f:
                first_line = f.readline().strip().split(",")
                recreate = first_line != header
    
        if recreate:
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(header)
    
    
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
        contains_patient_label = "Patient" in response  # case-sensitive match
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
        kl_loss = self.config.kl_coeff * (response_logprobs - old_logprobs).mean()
        total_loss = policy_loss + kl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.doctor_model.parameters(), 0.5)
        self.optimizer.step()

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
    
            queries, responses, norm_rewards, raw_rewards, diagnosis_scores, llm_diagnoses = training_data
    
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
            self.training_step_count += 1  # Increment only if training occurs
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
        all_queries, all_responses = [], []
        all_diag_scores, all_llm_diagnoses = [], []
    
        leaf_rewards, parent_rewards = [], []
        leaf_queries, parent_queries = [], []
        leaf_responses, parent_responses = [], []
        leaf_scores, parent_scores = [], []
        leaf_diagnoses, parent_diagnoses = [], []
    
        parent_acc = {}
    
        # ------- 1) handle leaf branches ----------------------------------------
        leaves = [b for b in branches if len(b) == self.config.num_turns]
        n_groups = len(leaves) // self.config.branches
    
        for g in range(n_groups):
            group = leaves[g * self.config.branches: (g + 1) * self.config.branches]
    
            for i, br in enumerate(group):
                full_txt = self._get_full_conversation_text(br)
                sibs = [self._get_full_conversation_text(b) for j, b in enumerate(group) if j != i]
                qry, rsp = self._split_query_response(br)
    
                if not self._is_valid_doctor_response(rsp):
                    diag_score = 0.0
                    llm_diag = "skipped"
                    print(f"⚠️ Invalid leaf doctor response: \"{rsp}\"")
                else:
                    llm_diag = self.reward_calculator.get_llm_diagnosis(full_txt)
                    diag_score = self.reward_calculator.compare_diagnoses(llm_diag, br[0].text)
    
                sibling_scores = [
                    self.reward_calculator.compare_diagnoses(
                        self.reward_calculator.get_llm_diagnosis(s), br[0].text
                    ) for s in sibs
                ]
                mean_sib_score = np.mean(sibling_scores) if sibling_scores else 0.0
                rel_reward = 0.5 + 0.5 * (diag_score - mean_sib_score)
    
                leaf_queries.append(qry)
                leaf_responses.append(rsp)
                leaf_rewards.append(rel_reward)
                leaf_scores.append(diag_score)
                leaf_diagnoses.append(llm_diag)
    
                parent_key = self._get_full_conversation_text(br[:-2])
                parent_acc.setdefault(parent_key, {"branch": br[:-2], "scores": []})["scores"].append(diag_score)
    
        # ------- 2) handle parent-level branches --------------------------------
        parent_means = {k: np.mean(v["scores"]) for k, v in parent_acc.items()}
    
        for k, info in parent_acc.items():
            my_mean = parent_means[k]
            sib_means = [v for kk, v in parent_means.items() if kk != k]
            sib_mean = np.mean(sib_means) if sib_means else my_mean
            rel_reward = 0.5 + 0.5 * (my_mean - sib_mean)
    
            qry, rsp = self._split_query_response(info["branch"])
            if not self._is_valid_doctor_response(rsp):
                diag_score = 0.0
                llm_diag = "skipped"
                print(f"⚠️ Invalid parent doctor response: \"{rsp}\"")
            else:
                convo_txt = self._get_full_conversation_text(info["branch"])
                llm_diag = self.reward_calculator.get_llm_diagnosis(convo_txt)
                diag_score = my_mean
    
            parent_queries.append(qry)
            parent_responses.append(rsp)
            parent_rewards.append(rel_reward)
            parent_scores.append(diag_score)
            parent_diagnoses.append(llm_diag)
    
        if not leaf_queries and not parent_queries:
            return None
    
        # ------- 3) check reward variance ---------------------------------------
        leaf_raw = torch.tensor(leaf_rewards, device=self.config.device).float() if leaf_rewards else torch.tensor([], device=self.config.device)
        parent_raw = torch.tensor(parent_rewards, device=self.config.device).float() if parent_rewards else torch.tensor([], device=self.config.device)
    
        leaf_std = leaf_raw.std().item() if leaf_raw.numel() > 1 else 0.0
        parent_std = parent_raw.std().item() if parent_raw.numel() > 1 else 0.0
    
        if leaf_std < 1e-6 and parent_std < 1e-6:
            print("⚠️ Skipping batch due to low reward variance (all branches scored equally)")
            return None
    
        # ------- 4) normalize separately ----------------------------------------
        if leaf_raw.numel() > 1 and leaf_std > 1e-6:
            norm_leaf = (leaf_raw - leaf_raw.mean()) / (leaf_std + 1e-8)
        else:
            norm_leaf = leaf_raw
    
        if parent_raw.numel() > 1 and parent_std > 1e-6:
            norm_parent = (parent_raw - parent_raw.mean()) / (parent_std + 1e-8)
        else:
            norm_parent = parent_raw
    
        all_queries = leaf_queries + parent_queries
        all_responses = leaf_responses + parent_responses
        all_rewards = torch.cat([norm_leaf, norm_parent])
        all_raw_rewards = torch.cat([leaf_raw, parent_raw])
        all_diag_scores = leaf_scores + parent_scores
        all_llm_diagnoses = leaf_diagnoses + parent_diagnoses
    
        # ------- 5) tokenize -----------------------------------------------------
        tok_args = dict(padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        queries_t = self.tokenizer(all_queries, **tok_args)["input_ids"].to(self.config.device)
        responses_t = self.tokenizer(all_responses, **tok_args)["input_ids"].to(self.config.device)
        diag_t = torch.tensor(all_diag_scores, device=self.config.device).float()
    
        return queries_t, responses_t, all_rewards, all_raw_rewards, diag_t, all_llm_diagnoses


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
        for i in range(16):
            if i < len(decoded_queries):
                convo = f"Patient: {decoded_queries[i].strip()}\nDoctor: {decoded_responses[i].strip()}"
                diag_score = diagnosis_scores[i].item()
                llm_diag   = llm_diagnoses[i]  # ✅ use directly from reward step
            else:
                convo, diag_score, llm_diag = "", 0.0, ""
            row.extend([convo, diag_score, llm_diag])
    
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)



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


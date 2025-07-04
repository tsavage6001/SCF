from typing import Dict, List, Tuple
import torch
import pandas as pd
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass

import os
from openai import OpenAI

import os
import torch
import pandas as pd
from dataclasses import dataclass
from openai import OpenAI
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

@dataclass
class ConversationTurn:
    speaker: str
    text: str
    input_ids: torch.Tensor
    response_ids: torch.Tensor

from accelerate import Accelerator

class ConversationGenerator:
    def __init__(
        self,
        config,
        tokenizer: AutoTokenizer,
        patient_model: AutoModelForCausalLM,
        doctor_model: AutoModelForCausalLM,
        medical_data: pd.DataFrame
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.patient_model = patient_model
        self.doctor_model = doctor_model
        self.medical_data = medical_data

        self.accelerator = Accelerator()  # ✅ Add this line

        # pad/eos config
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.doctor_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.doctor_model.config.eos_token_id = self.tokenizer.eos_token_id

        # prepare models with accelerate
        self.doctor_model, self.tokenizer = self.accelerator.prepare(self.doctor_model, self.tokenizer)

        self._setup_generation_config()

        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    def _setup_generation_config(self):
        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            num_return_sequences=self.config.branches
        )

    def generate_turn(self, prompt: str, speaker: str) -> ConversationTurn:
        """Generate a single conversation turn"""
        model = self.patient_model if speaker == "Patient" else self.doctor_model
        use_grad = speaker != "Patient"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        
        with torch.set_grad_enabled(use_grad):
            outputs = model.generate(
                **inputs,
                generation_config=self.generation_config
            )

        return ConversationTurn(
            speaker=speaker,
            text=self._decode_response(outputs, inputs),
            input_ids=inputs.input_ids[0],
            response_ids=outputs[0]
        )

    def _decode_response(self, outputs, inputs) -> str:
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )

    def generate_branched_conversation(self, patient_input: str) -> List[List[ConversationTurn]]:
        """Generate complete branched conversation with only doctor turns branching."""
        initial_input = self.tokenizer(f"Patient: {patient_input}", return_tensors="pt")
        initial_input_ids = initial_input["input_ids"].to(self.config.device)
    
        # Start with a single branch containing the patient's first message
        branches = [[self._create_initial_turn(patient_input, initial_input_ids)]]
    
        for turn_idx in range(self.config.num_turns - 1):
            current_speaker = "Doctor" if turn_idx % 2 == 0 else "Patient"
            branches = self._process_turn(branches, current_speaker, turn_idx)
    
            # Debug: Monitor how many branches exist after each turn
            print(f"[DEBUG] After turn {turn_idx + 1}: {len(branches)} branches")
    
        return branches

    def _create_initial_turn(self, text: str, input_ids: torch.Tensor) -> ConversationTurn:
        return ConversationTurn(
            speaker="Patient",
            text=text,
            input_ids=input_ids[0],
            response_ids=input_ids[0]
        )

    def _process_turn(
        self,
        branches: List[List[ConversationTurn]],
        speaker: str,
        turn_idx: int
    ) -> List[List[ConversationTurn]]:
        """Process next turn, branching only at doctor steps."""
        new_branches = []
    
        for branch_idx, branch in enumerate(branches):
            if speaker == "Doctor":
                # Doctor turn: branch into multiple options
                generated_doctor_branches = self._generate_doctor_turns(branch, branch_idx)
                new_branches.extend(generated_doctor_branches)
            else:
                # Patient turn: only one continuation, no branching
                continued_branch = self._generate_patient_turn(branch)
                new_branches.append(continued_branch)
    
        return new_branches

    def _build_doctor_prompt(self, branch: List[ConversationTurn], history_window: int = 5) -> str:
        system_message = "You are an experienced doctor performing a medical interview of the patient. Strictly answer in less than 10 words. DO NOT enter a response as a patient.\n"
        instruction = "Read the above patient complaint/conversation and respond with a single question in strictly less than 10 words.\n"
    
        # Only keep last N conversation turns for context
        truncated_branch = branch[-history_window:]
    
        history = "\n".join(f"{turn.speaker}: {turn.text}" for turn in truncated_branch)
        return f"{system_message}\n{instruction}\nCurrent Conversation: {history}\nDoctor (your response):"

    def _build_patient_prompt(self, branch: List[ConversationTurn], case_idx: int) -> str:
        system_message = """You are a patient at a doctor’s office responding to a doctor’s questions for the given patient scenario.  Answer in less than 20 words.  Do NOT provide additional information (from the Additional context) unless explicitly asked by the doctor.  If there is no question from the doctor, respond with "silence".  Do NOT enter a response as a physician.\n"""
        details = self.medical_data.iloc[case_idx]['details']
        history = "\n".join(f"{turn.speaker}: {turn.text}" for turn in branch)
        return f"{system_message} \n Additional context: {details} \n Current Conversation: {history} \n Patient (your response): "


    def _build_prompt(self, branch: List[ConversationTurn], case_idx: int = None) -> str:
        current_speaker = "Doctor" if branch[-1].speaker == "Patient" else "Patient"
        if current_speaker == "Doctor":
            return self._build_doctor_prompt(branch)
        else:
            return self._build_patient_prompt(branch, case_idx)

    def _generate_doctor_turns(
        self,
        branch: List[ConversationTurn],
        branch_idx: int
    ) -> List[List[ConversationTurn]]:
        # 🔄 Prompt setup
        system_message = "You are an experienced doctor performing a medical interview of the patient. Answer in less than 10 words. DO NOT enter a response as a patient."
        instruction = "Read the above patient complaint/conversation and respond with a single question less than 10 words."
        history = "\n".join(f"{turn.speaker}: {turn.text}" for turn in branch)
        prompt = f"{system_message}\n{instruction}\n{history}\nDoctor:"
    
        #print("\n🧠 [Doctor Prompt being sent to model]:\n" + "-"*40)
        #print(prompt)
        #print("-" * 40)
    
        self.doctor_model.eval()
    
        # ✅ Tokenize and send to same device as model
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.doctor_model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
    
        #print(f"🔍 Prompt input token count: {input_len}")
    
        repeated_inputs = {
            "input_ids": inputs["input_ids"].repeat(self.config.branches, 1),
            "attention_mask": inputs["attention_mask"].repeat(self.config.branches, 1)
        }
    
        # Prevent copying "Patient:" or "Doctor:" in outputs
        bad_words_ids = self.tokenizer(["Patient:", "Doctor:"], add_special_tokens=False).input_ids
    
        with torch.no_grad():
            outputs = self.doctor_model.generate(
                **repeated_inputs,
                max_new_tokens=self.config.max_length,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False,
                bad_words_ids=bad_words_ids,
                return_dict_in_generate=True
            )
    
       # print(f"📦 Generated sequences shape: {outputs.sequences.shape}")
    
        new_branches = []
        for i in range(self.config.branches):
            full_output = outputs.sequences[i]
            generated_tokens = full_output[input_len:]
    
          #  print(f"🧪 Raw token ids (branch {i+1}): {generated_tokens.tolist()}")
    
            # Inside _generate_doctor_turns loop
            decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            if not decoded or decoded.lower() in ["", "i don't know", "not sure"]:
                print(f"⚠️ Degenerate output detected on branch {i+1}: '{decoded}'")
                continue  # ⛔️ Skip adding this branch entirely OR regenerate with fallback sampling
            
    
        #    print(f"🗣️ Doctor response (branch {i+1}): '{decoded}'")
    
            turn = ConversationTurn(
                speaker="Doctor",
                text=decoded,
                input_ids=inputs["input_ids"][0],
                response_ids=full_output
            )
            new_branch = branch.copy()
            new_branch.append(turn)
            new_branches.append(new_branch)
    
      #      self._print_conversation(new_branch, branch_idx * self.config.branches + i + 1)
    
        return new_branches


    


    def _generate_patient_turn(self, branch: List[ConversationTurn]) -> List[ConversationTurn]:
        # Determine case index from the initial patient input
        initial_case = branch[0].text
        case_idx = self.medical_data[self.medical_data['case'] == initial_case].index[0]
        
        prompt = self._build_prompt(branch, case_idx)
     #   print("\n💬 [Patient Prompt being sent to GPT-4o API]:\n" + "-"*40)
     #   print(prompt)
     #   print("-" * 40)
    
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=32,
            )
            text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ GPT-4o API Error: {str(e)}")
            text = "[No response]"
    
        # Tokenize for record keeping
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        response_ids = self.tokenizer(text, return_tensors="pt")["input_ids"][0]
    
        turn = ConversationTurn(
            speaker="Patient",
            text=text,
            input_ids=input_ids,
            response_ids=response_ids
        )
        new_branch = branch.copy()
        new_branch.append(turn)
    #    self._print_conversation(new_branch, len(new_branch))
        return new_branch

    def _print_conversation(self, branch: List[ConversationTurn], branch_num: int):
        # Define ANSI color codes
        COLOR_RESET = "\033[0m"
        COLOR_DOCTOR = "\033[94m"   # Blue
        COLOR_PATIENT = "\033[92m"  # Green
    
    #    print("\n" + "=" * 50)
    #    print(f"🌿 Branch {branch_num}")
    #    print("=" * 50)
        for i, turn in enumerate(branch):
            color = COLOR_DOCTOR if turn.speaker == "Doctor" else COLOR_PATIENT
      #      print(f"[Turn {i+1}] {turn.speaker}: {color}{turn.text}{COLOR_RESET}")
    

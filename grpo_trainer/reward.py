import numpy as np
import pandas as pd
from typing import List, Dict
from openai import OpenAI
import hashlib

class RewardCalculator:
    def __init__(self):
        self.client = OpenAI(api_key='KEY')
        self.diagnosis_cache = {}
        self.score_cache = {}

        self.medical_data = pd.read_csv("training_cases.csv")


    def _digest(self, text: str) -> str:
        """Return a short hash of the text for display."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:6]
        
    def get_llm_diagnosis(self, conversation: str) -> str:
        """Cache and get diagnosis from GPT-4 based on conversation"""
        if conversation in self.diagnosis_cache:
            return self.diagnosis_cache[conversation]

        prompt = f"""Based on the following doctor-patient conversation, 
what is the most likely diagnosis? Provide only the diagnosis name.

Conversation:
{conversation}

Diagnosis:"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        diagnosis = response.choices[0].message.content.strip().lower()
        print(f"🔎 LLM Diagnosis: {diagnosis}")
        self.diagnosis_cache[conversation] = diagnosis
        return diagnosis

    def compare_diagnoses(self, llm_diagnosis: str, case: str) -> float:
        """Compare LLM diagnosis with expected and cache scores"""
        key = (llm_diagnosis, case)
        if key in self.score_cache:
            return self.score_cache[key]

        try:
            matched = self.medical_data[
                self.medical_data['case'].str.lower() == case.lower()
            ]
            if matched.empty:
                print(f"⚠️ No match found for case: {case[:50]}")
                return 0.0

            expected = matched['diagnosis'].iloc[0].lower()
            if llm_diagnosis == expected:
                self.score_cache[key] = 1.0
                return 1.0

            prompt = f"""Compare these medical diagnoses. Score from 0-1 where:
1 = Exact match or clinically equivalent
0.5 = Related but different severity
0 = Completely different

Diagnosis 1: {llm_diagnosis}
Diagnosis 2: {expected}

Provide only the numerical score:"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            score = float(response.choices[0].message.content.strip())
            score = min(max(score, 0.0), 1.0)
            print(f"🧮 Diagnosis Score ({llm_diagnosis} vs {expected}): {score}")
            self.score_cache[key] = score
            return score

        except Exception as e:
            print(f"⚠️ Error comparing diagnoses: {str(e)}")
            return 0.0

    
    

    def get_cached_diagnosis(self, conversation: str) -> str:
        """Get cached LLM diagnosis (if available), else re-run"""
        return self.get_llm_diagnosis(conversation)
    
        
    def calculate_reward(
        self,
        doctor_response: str,
        sibling_responses: List[str],
        full_conversation: str,
        initial_case: str,
        branch_id: int = 0  # optional: passed in by outer loop for clarity
    ) -> float:
        """Calculate reward incorporating diagnosis accuracy."""
        conv_hash = self._digest(full_conversation)

        print(f"\n🔎 [Branch {branch_id}] Conversation Hash: {conv_hash}")
        print(f"📋 Full Conversation:\n{full_conversation}\n")
    
        llm_diagnosis = self.get_llm_diagnosis(full_conversation)
        diagnosis_score = self.compare_diagnoses(llm_diagnosis, initial_case)
    
        print(f"🧠 Branch {branch_id} LLM Diagnosis: {llm_diagnosis}")
        print(f"📈 Branch {branch_id} Diagnosis Score: {diagnosis_score:.3f}")
    
        sibling_scores = []
        for i, sibling_conv in enumerate(sibling_responses):
            sib_hash = self._digest(sibling_conv)
            sib_diag = self.get_llm_diagnosis(sibling_conv)
            sib_score = self.compare_diagnoses(sib_diag, initial_case)
            sibling_scores.append(sib_score)
            print(f"   ↳ Sibling {i+1} [{sib_hash}] Diagnosis: {sib_diag} (Score: {sib_score:.3f})")
    
        if not sibling_scores:
            relative_score = diagnosis_score
            mean_sibling_score = 0
        else:
            mean_sibling_score = np.mean(sibling_scores)
            relative_score = 0.5 + 0.5 * (diagnosis_score - mean_sibling_score)
    
        print(f"🏅 Branch {branch_id} Final Reward: {relative_score:.3f} "
              f"(Diagnosis: {diagnosis_score:.3f}, Siblings Avg: {mean_sibling_score:.3f})")
    
        return relative_score, diagnosis_score


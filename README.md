Savage Conversation Forests (SCF)
Fine-Tuning Large Language Models for Multi-Turn Medical Interviews with Branching Reinforcement Learning

Overview
Savage Conversation Forests (SCF) is a novel reinforcement learning framework designed to fine-tune large language models (LLMs) for multi-turn medical conversations. Unlike existing methods (e.g., PPO, GRPO, DPO) that operate on isolated single-turn prompts, SCF enables inter-turn learning by structuring dialogues as branching conversation trees. This allows LLMs to learn diagnostic strategies and understand how each conversational turn influences future conversation flow.

SCF is built on top of Group Relative Policy Optimization (GRPO) and introduces three key innovations:

Branching Multi-Turn Architecture: Each doctor turn spawns multiple possible responses, forming a tree of dialogue trajectories.

Sibling-Relative Reward Calculation: Each completion is evaluated relative to its siblings at the same conversational depth.

Depth-Wise Normalization: Training signals are normalized separately for parent and leaf nodes to ensure balanced reward propagation.


=============

To use the code you will need to download requirements in the requirement.txt file, export your OpenAI API key, and sign into Huggingface.

"""
PEFT fine-tuning for Guardian models using QLoRA.

This module provides Parameter-Efficient Fine-Tuning (PEFT) capabilities for Guardian
models using QLoRA (Quantized Low-Rank Adaptation). It supports fine-tuning of all
three model types (summarizer, extractor, weak_labeler) on synthetic case data.

The fine-tuning system is designed to improve model performance on Guardian-specific
tasks while maintaining efficiency through quantization and low-rank adaptation.

Author: Guardian AI System


Classes:
    GuardianFineTuner: Main fine-tuning class with QLoRA support

Functions:
    fine_tune_summarizer(): Fine-tune summarizer model
    fine_tune_extractor(): Fine-tune extractor model  
    fine_tune_weak_labeler(): Fine-tune weak labeler model

Example:
    >>> from llm import fine_tune_summarizer
    >>> trainer = fine_tune_summarizer(train_data, eval_data)
    >>> # Model is saved to ./finetuned_summarizer/
"""
import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from .prompts import FINE_TUNE_SUMMARY_PROMPT, FINE_TUNE_EXTRACTION_PROMPT, FINE_TUNE_LABELING_PROMPT

class GuardianFineTuner:
    """
    Fine-tuning class for Guardian models using QLoRA.
    
    This class provides a comprehensive fine-tuning interface for Guardian models
    using Parameter-Efficient Fine-Tuning (PEFT) with QLoRA. It supports all three
    Guardian model types and provides optimized training configurations.
    
    Attributes:
        model_path (str): Path to the base model for fine-tuning
        task_type (str): Type of task ("summarizer", "extractor", "weak_labeler")
        tokenizer: Hugging Face tokenizer for the model
        model: Base model for fine-tuning
        peft_model: PEFT-wrapped model with LoRA adapters
        
    Example:
        >>> tuner = GuardianFineTuner("./models/Llama3_1-8B-Instruct", "summarizer")
        >>> tuner.load_model()
        >>> tuner.setup_lora()
        >>> trainer = tuner.train(train_data, eval_data)
    """
    
    def __init__(self, model_path: str, task_type: str = "summarizer"):
        """
        Initialize the GuardianFineTuner.
        
        Args:
            model_path (str): Path to the base model directory
            task_type (str): Type of task for fine-tuning. Options:
                - "summarizer": For case summarization tasks
                - "extractor": For entity extraction tasks  
                - "weak_labeler": For movement/risk classification tasks
        """
        self.model_path = model_path
        self.task_type = task_type
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def load_model(self):
        """
        Load model and tokenizer for fine-tuning.
        
        This method loads the base model and tokenizer from the specified path,
        configuring them appropriately for fine-tuning. It handles device placement
        and data type optimization based on available hardware.
        
        Note:
            Must be called before setup_lora() and train().
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
    def setup_lora(self, r=16, lora_alpha=32, lora_dropout=0.1):
        """
        Setup LoRA configuration for parameter-efficient fine-tuning.
        
        This method configures Low-Rank Adaptation (LoRA) parameters for efficient
        fine-tuning. LoRA allows fine-tuning with significantly fewer parameters
        while maintaining performance.
        
        Args:
            r (int): Rank of the adaptation. Higher values = more parameters.
                     Default: 16 (good balance of performance and efficiency)
            lora_alpha (int): LoRA scaling parameter. Default: 32
            lora_dropout (float): Dropout rate for LoRA layers. Default: 0.1
            
        Note:
            Must be called after load_model() and before train().
            Prints trainable parameter count for verification.
        """
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
    def prepare_data(self, data: list):
        """
        Prepare training data for fine-tuning.
        
        This method formats the training data according to the task type,
        applying appropriate prompt templates and tokenization.
        
        Args:
            data (list): List of training examples. Each example should be a dict
                        with keys appropriate to the task type:
                        - summarizer: {"narrative": str, "summary": str}
                        - extractor: {"narrative": str, "extraction": str}
                        - weak_labeler: {"narrative": str, "movement": str, "risk": str}
        
        Returns:
            Dataset: Hugging Face Dataset object ready for training
        """
        if self.task_type == "summarizer":
            texts = [FINE_TUNE_SUMMARY_PROMPT.format(narrative=d["narrative"], summary=d["summary"]) for d in data]
        elif self.task_type == "extractor":
            texts = [FINE_TUNE_EXTRACTION_PROMPT.format(narrative=d["narrative"], extraction=d["extraction"]) for d in data]
        elif self.task_type == "weak_labeler":
            texts = [FINE_TUNE_LABELING_PROMPT.format(
                narrative=d["narrative"], 
                movement=d["movement"], 
                risk=d["risk"]
            ) for d in data]
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
            
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        return Dataset.from_dict(tokenized)
        
    def train(self, train_data: list, eval_data: list = None, 
              output_dir: str = "./finetuned_model", 
              num_epochs: int = 3, batch_size: int = 4):
        """
        Train the model using QLoRA fine-tuning.
        
        This method performs the actual fine-tuning process using the prepared
        training data and LoRA configuration. It handles training arguments,
        data collation, and model saving.
        
        Args:
            train_data (list): Training data examples
            eval_data (list, optional): Evaluation data examples
            output_dir (str): Directory to save the fine-tuned model
            num_epochs (int): Number of training epochs
            batch_size (int): Training batch size
            
        Returns:
            Trainer: Hugging Face Trainer object for further use
            
        Raises:
            ValueError: If LoRA not setup (call setup_lora() first)
        """
        if not self.peft_model:
            raise ValueError("LoRA not setup. Call setup_lora() first.")
            
        # Prepare data
        train_dataset = self.prepare_data(train_data)
        eval_dataset = self.prepare_data(eval_data) if eval_data else None
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="steps" if eval_data else "no",
            eval_steps=100 if eval_data else None,
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True if eval_data else False,
            metric_for_best_model="eval_loss" if eval_data else None,
            greater_is_better=False if eval_data else None,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer

def fine_tune_summarizer(train_data: list, eval_data: list = None, 
                        model_path: str = None, output_dir: str = "./finetuned_summarizer"):
    """
    Fine-tune summarizer model using QLoRA.
    
    Convenience function for fine-tuning the summarizer model on case data.
    Automatically loads the model from configuration and sets up QLoRA training.
    
    Args:
        train_data (list): Training examples with format:
            [{"narrative": str, "summary": str}, ...]
        eval_data (list, optional): Evaluation examples with same format
        model_path (str, optional): Path to base model. If None, uses config
        output_dir (str): Directory to save fine-tuned model
        
    Returns:
        Trainer: Hugging Face Trainer object
        
    Example:
        >>> train_data = [{"narrative": "Case text...", "summary": "Summary..."}]
        >>> trainer = fine_tune_summarizer(train_data)
    """
    if not model_path:
        CFG = json.load(open("guardian.config.json", "r"))
        model_path = CFG["models"]["summarizer_instruct"]
    
    tuner = GuardianFineTuner(model_path, "summarizer")
    tuner.load_model()
    tuner.setup_lora()
    return tuner.train(train_data, eval_data, output_dir)

def fine_tune_extractor(train_data: list, eval_data: list = None,
                       model_path: str = None, output_dir: str = "./finetuned_extractor"):
    """
    Fine-tune extractor model using QLoRA.
    
    Convenience function for fine-tuning the extractor model on entity extraction data.
    Automatically loads the model from configuration and sets up QLoRA training.
    
    Args:
        train_data (list): Training examples with format:
            [{"narrative": str, "extraction": str}, ...]
        eval_data (list, optional): Evaluation examples with same format
        model_path (str, optional): Path to base model. If None, uses config
        output_dir (str): Directory to save fine-tuned model
        
    Returns:
        Trainer: Hugging Face Trainer object
        
    Example:
        >>> train_data = [{"narrative": "Case text...", "extraction": "JSON entities..."}]
        >>> trainer = fine_tune_extractor(train_data)
    """
    if not model_path:
        CFG = json.load(open("guardian.config.json", "r"))
        model_path = CFG["models"]["extractor"]
    
    tuner = GuardianFineTuner(model_path, "extractor")
    tuner.load_model()
    tuner.setup_lora()
    return tuner.train(train_data, eval_data, output_dir)

def fine_tune_weak_labeler(train_data: list, eval_data: list = None,
                          model_path: str = None, output_dir: str = "./finetuned_weak_labeler"):
    """
    Fine-tune weak labeler model using QLoRA.
    
    Convenience function for fine-tuning the weak labeler model on movement/risk data.
    Automatically loads the model from configuration and sets up QLoRA training.
    
    Args:
        train_data (list): Training examples with format:
            [{"narrative": str, "movement": str, "risk": str}, ...]
        eval_data (list, optional): Evaluation examples with same format
        model_path (str, optional): Path to base model. If None, uses config
        output_dir (str): Directory to save fine-tuned model
        
    Returns:
        Trainer: Hugging Face Trainer object
        
    Example:
        >>> train_data = [{"narrative": "Case text...", "movement": "Local", "risk": "High"}]
        >>> trainer = fine_tune_weak_labeler(train_data)
    """
    if not model_path:
        CFG = json.load(open("guardian.config.json", "r"))
        model_path = CFG["models"]["weak_labeler"]
    
    tuner = GuardianFineTuner(model_path, "weak_labeler")
    tuner.load_model()
    tuner.setup_lora()
    return tuner.train(train_data, eval_data, output_dir)

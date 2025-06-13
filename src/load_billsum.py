from datasets import load_dataset
import logging
from typing import Dict, Optional, Union
from datasets import Dataset, DatasetDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_billsum_dataset(
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42
) -> Union[Dataset, DatasetDict]:
    
    try:
        logger.info(f"Loading BillSum dataset{f' ({split} split)' if split else ''}")
        
        # Load the dataset
        dataset = load_dataset("FiscalNote/billsum", split=split)
        
        if shuffle:
            logger.info("Shuffling dataset")
            if isinstance(dataset, DatasetDict):
                dataset = DatasetDict({
                    k: v.shuffle(seed=seed) for k, v in dataset.items()
                })
            else:
                dataset = dataset.shuffle(seed=seed)
        
        if max_samples:
            logger.info(f"Limiting to {max_samples} samples")
            if isinstance(dataset, DatasetDict):
                dataset = DatasetDict({
                    k: v.select(range(min(max_samples, len(v))))
                    for k, v in dataset.items()
                })
            else:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # Log dataset statistics
        if isinstance(dataset, DatasetDict):
            for split_name, split_data in dataset.items():
                logger.info(f"{split_name} split size: {len(split_data)}")
        else:
            logger.info(f"Dataset size: {len(dataset)}")
        
        return dataset
    
    except Exception as e:
        logger.error(f"Error loading BillSum dataset: {str(e)}")
        raise

def get_sample(
    dataset: Union[Dataset, DatasetDict],
    index: int = 0,
    split: str = "train"
) -> Dict:
    
    try:
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise ValueError(f"Split '{split}' not found in dataset")
            data = dataset[split]
        else:
            data = dataset
        
        if index >= len(data):
            raise ValueError(f"Index {index} out of range for dataset of size {len(data)}")
        
        sample = data[index]
        return {
            "text": sample["text"],
            "summary": sample["summary"]
        }
    
    except Exception as e:
        logger.error(f"Error getting sample from dataset: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        # Load all splits
        full_dataset = load_billsum_dataset()
        print("\nFull dataset loaded successfully!")
        
        # Load just the training split
        train_dataset = load_billsum_dataset(split="train", max_samples=5, shuffle=True)
        print("\nTraining subset loaded successfully!")
        
        # Get a sample
        sample = get_sample(train_dataset)
        print("\nSample summary length:", len(sample["summary"]))
        
    except Exception as e:
        print(f"Error in example: {str(e)}")
from imports_104 import *
from utils_104 import *

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, default="./data/2wikimultihopqa/train_2passages_deduplication_0_300.json")
    args.add_argument("--model_path", type=str, default="./models/Llama-3.2-1B-Instruct-Doc_mask")
    args.add_argument("--device", type=str, default="cuda:6")
    args = args.parse_args()
    # dataset_paths=[
    #     "./data_aug_projector/2wikimultihopqa/llama3.2-1b-instruct/bridge_comparison.json",
    #     "./data_aug_projector/2wikimultihopqa/llama3.2-1b-instruct/comparison.json",
    #     "./data_aug_projector/2wikimultihopqa/llama3.2-1b-instruct/compositional.json",
    #     "./data_aug_projector/2wikimultihopqa/llama3.2-1b-instruct/inference.json",
    #     "./data_aug_projector/complexwebquestions/llama3.2-1b-instruct/total.json",
    #     "./data_aug_projector/hotpotqa/llama3.2-1b-instruct/bridge.json",
    #     "./data_aug_projector/hotpotqa/llama3.2-1b-instruct/comparison.json",
    #     "./data_aug_projector/popqa/llama3.2-1b-instruct/total.json"
    # ]
    # output_base_dir="./data_aug_deepseek-v3"
    # reaugment_with_deepseek(dataset_paths, output_base_dir)

    # dataset_paths=[
    #     "./data/2wikimultihopqa/train_passages_deduplication_30000.json"
    # ]
    dataset_paths = [
        args.dataset_path
    ]
    # output_base_dir="./data_aug_deepseek-v3"
    # augment_with_deepseek_multipassage(dataset_paths, output_base_dir)
    llm_model_name = Path(args.model_path).parts[-1]
    output_base_dir = f"./data_aug_{llm_model_name}"
    augment_with_model(dataset_paths, output_base_dir, args.model_path, max_new_tokens=2048, device=args.device)

    # dataset_paths = [
    #     "./data_aug/2wikimultihopqa/llama3.2-1b-instruct/bridge_comparison.json",
    #     "./data_aug/2wikimultihopqa/llama3.2-1b-instruct/comparison.json",
    #     "./data_aug/2wikimultihopqa/llama3.2-1b-instruct/compositional.json",
    #     "./data_aug/2wikimultihopqa/llama3.2-1b-instruct/inference.json",
    #     "./data_aug/complexwebquestions/llama3.2-1b-instruct/total.json",
    #     "./data_aug/hotpotqa/llama3.2-1b-instruct/bridge.json",
    #     "./data_aug/hotpotqa/llama3.2-1b-instruct/comparison.json",
    #     "./data_aug/popqa/llama3.2-1b-instruct/total.json",
    #     "./data_aug/iirc/llama3-8b-instruct/total.json",
    #     "./data_aug/ragtruth/llama3-8b-instruct/total.json",
    #     "./data_aug/strategyqa/llama3-8b-instruct/total.json"
    # ]
    # output_base_dir = "./data_dev"
    # strip_augment_field(dataset_paths, output_base_dir)

    # dataset_paths=[
    #     "./data_aug_deepseek-v3/2wikimultihopqa/llama3.2-1b-instruct/bridge_comparison.json",
    #     "./data_aug_deepseek-v3/2wikimultihopqa/llama3.2-1b-instruct/comparison.json",
    #     "./data_aug_deepseek-v3/2wikimultihopqa/llama3.2-1b-instruct/compositional.json",
    #     "./data_aug_deepseek-v3/2wikimultihopqa/llama3.2-1b-instruct/inference.json",
    #     "./data_aug_deepseek-v3/complexwebquestions/llama3.2-1b-instruct/total.json",
    #     "./data_aug_deepseek-v3/hotpotqa/llama3.2-1b-instruct/bridge.json",
    #     "./data_aug_deepseek-v3/hotpotqa/llama3.2-1b-instruct/comparison.json",
    #     "./data_aug_deepseek-v3/popqa/llama3.2-1b-instruct/total.json"
    # ]

    # inference_on_dataset("./data_aug_deepseek-v3/train", "./models/Llama-3.2-1B-Instruct", 256, 4)
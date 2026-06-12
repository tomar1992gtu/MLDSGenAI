import argparse

from pipelines.training_pipeline import TrainingPipeline
from pipelines.inference_pipeline import InferencePipeline


parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    choices=["train", "predict"],
    required=True
)

parser.add_argument(
    "--config",
    required=True
)

args = parser.parse_args()

if args.mode == "train":

    pipeline = TrainingPipeline(
        config_path=args.config
    )

    print(pipeline.run())

elif args.mode == "predict":

    sample = {
        # test data
    }

    print(
        InferencePipeline(
            config_path=args.config
        ).run(sample)
    )


'''
python main.py \
    --mode predict \
    --config configs/customer_churn_params.yaml \
    --input input.json
'''

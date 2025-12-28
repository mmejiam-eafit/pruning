from sagemaker.pytorch import PyTorch

SECONDS_IN_HOUR = 60*60
HOURS = 86
estimator = PyTorch(
    entry_point="main_sagemaker.py",
    source_dir="",
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    volume_size=92,
    role="SageMakerRole",
    max_run=HOURS * SECONDS_IN_HOUR,
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.12.0-gpu-py38-cu116-ubuntu20.04-e3-v1.0",

)

estimator.fit(inputs={
    "dataset": "s3://chexnet-dataset-divisions",
    "database": "s3://chexnet-dataset-images"
})
def lambda_handler(event, context):
    from src.training.script import train_and_upload  ← inside function
    acc = train_and_upload()

    return {
        "statusCode": 200,
        "message": "Model trained and uploaded to S3",
        "accuracy": acc
    }

from ultralytics import YOLO

def main():

    print("Loading model...")

    # Load the model.
    model = YOLO(
        input(
            "Enter the starting model for training, either a default model (yolo11n.pt) or a previus ending point (your_model.pt): "
        )
    )  # change to last trained model or leave it as it is to train from scratch

    model_name = input("Enter the name of the model: ")

    enable_clearml = bool(input("Enable ClearML(tool for tracking model progress, only for advanced users)? (True/False): "))
    if enable_clearml:
        from clearml import Task

        Task.init(project_name='FRC_object_detection', task_name=model_name)

    print("Model loaded.")
    print("Starting training...")

    # Training.
    model.train(
        data=input(
            "Input dataset path, recommend using roboflow: "
        ),  # change dataset path, this is from roboflow
        imgsz=int(input("Enter the image size, 320 is a good starting point: ")),  # change to change
        epochs=int(
            input(
                "Enter the number of epoches you want to train for, 180 is a larger number for this: "
            )
        ),  # edit to change how long you will train
        name=model_name,
        device=input(
            "Enter what device to train with, 0 is first gpu, cpu is for cpu, and an array of numbers for multi gpu: "
        ),  # 0 is the first GPU, remove this line to use CPU, use array to use multiple GPUs, GPU is much faster
        patience=int(
            input(
                "Enter the patience value, 20 is a good starting point: "
            )
        ),  # change to change confidence threshold
    )

    print("Training complete.")


if __name__ == "__main__":
    main()

"""app-pytorch: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
import time
from pathlib import Path
from flower_rise.task import Net, load_data
from flower_rise.task import test as test_fn
from flower_rise.task import train as train_fn
import csv

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    total_start_time = time.time()
        #start keeping record

    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        # msg.content["config"]["momentum"],
        device,
    )
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    

    
    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "time_train_record":total_duration,
        "num-examples": len(trainloader.dataset),
    }


    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    total_start_time = time.time()

    eval_csv_path = Path(f"Fed_records/ten_rounds/client_eval.csv")
    eval_csv_path.parent.mkdir(parents=True, exist_ok=True)

    """Evaluate the model on local data."""
    file_exists=eval_csv_path.exists()
    partition_id = context.node_config["partition-id"]

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)



    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"Total wall-clock time: {total_duration:.2f} seconds")
    

    



    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "time_evaluate_record":total_duration,
        "total_duration":total_duration,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
"""app-pytorch: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
import time
from pathlib import Path
from flower_rise.task_Fedprox import Net, load_data
from flower_rise.task_Fedprox import test as test_fn
from flower_rise.task_Fedprox import train as train_fn
import csv
import copy

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    total_start_time = time.time()
        #start keeping record

#which distribution to choose
    distribution_type = context.run_config.get("data-distribution", "iid")
    alpha = context.run_config.get("dirichlet-alpha", 0.5)
    num_classes = context.run_config.get("pathological-classes", 2)

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    global_params = [p.clone().detach() for p in model.parameters()]


    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions
        , distribution=distribution_type,
        alpha=alpha,
        num_classes_per_client=num_classes,
)

    local_epochs = context.run_config["local-epochs"]
    lr = msg.content["config"]["lr"]
    momentum = msg.content["config"]["momentum"]
    mu = msg.content["config"].get("mu", 0.0)  # Get mu from config, default 0

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        local_epochs,
        lr,
        momentum,
        device,
        mu=mu,
        global_params=global_params,
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
    #which distribution to choose
    distribution_type = context.run_config.get("data-distribution", "iid")
    alpha = context.run_config.get("dirichlet-alpha", 0.5)
    num_classes = context.run_config.get("pathological-classes", 2)

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
    _, valloader = load_data(partition_id, num_partitions
        , distribution=distribution_type,
        alpha=alpha,
        num_classes_per_client=num_classes,)



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
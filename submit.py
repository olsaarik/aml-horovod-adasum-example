from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.container_registry import ContainerRegistry
from azureml.core import Workspace, Experiment, runconfig
from azureml.train.estimator import Estimator

ws = Workspace.from_config("config_aml.json")

experiment_name = 'MNIST'
experiment = Experiment(ws, name=experiment_name)

gpu_cluster_name = "NC24"
try:
    gpu_compute_target = ComputeTarget(workspace=ws, name=gpu_cluster_name)
    print('Found existing compute target.')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(
        vm_size='Standard_NC24',
        min_nodes=0,
        max_nodes=8,
    )
    gpu_compute_target = ComputeTarget.create(ws, gpu_cluster_name, compute_config)
    gpu_compute_target.wait_for_completion(show_output=True)

mpi_config=runconfig.MpiConfiguration()
mpi_config.process_count_per_node = 4

docker_env=runconfig.DockerSection()
docker_env.enabled=True

cd = runconfig.CondaDependencies()
cd.add_channel('pytorch')
cd.add_conda_package('pytorch')
cd.add_conda_package('torchvision')
cd.add_pip_package('horovod-adasum')

env_def=runconfig.EnvironmentDefinition()
env_def.docker=docker_env
env_def.python.conda_dependencies = cd

estimator = Estimator(
    source_directory='.',
    compute_target=gpu_compute_target,
    node_count=8,
    distributed_training=mpi_config,
    environment_definition=env_def,
    entry_script='pytorch_mnist.py',
)

run = experiment.submit(estimator)
print(run.get_portal_url())
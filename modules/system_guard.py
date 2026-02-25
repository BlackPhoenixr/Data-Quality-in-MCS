"""
This module provides the following utilities:
    - Imports and Dependencies: Importing necessary libraries and modules.
    - Logger Setup: A custom logger configuration using Loguru for consistent logging.
    - Timer Decorator: A decorator to measure the execution time of functions.
    - Signal Handler: A function to handle signals and trigger cleanup operations.
    - Resource Cleanup: A function to clean up resources such as child processes and semaphores.
    - Active Children Cleanup: A function to remove orphaned semaphores and terminate resource tracker processes.
    - Memory Limit Calculation: A function to calculate the memory limit for Dask clusters based on available memory.
    - Create Dask Client: A function to create a Dask client connected to a local cluster.
    - Clean Project Outputs: A function to clean up selected subdirectories inside the outputs folder.
"""

# =========== Imports and Dependencies ===========
import sys  # Required for writing logs to stdout
from functools import wraps  # Preserves function metadata in decorators
from typing import Any  # Type hinting for any return type
from loguru import logger  # Import Loguru's logger
from multiprocessing import (
    active_children,
)  # Manages active child processes to ensure proper cleanup.
import gc  # Enables manual garbage collection to free up memory.
import os  # Provides a portable way of using operating system-dependent functionality.
import shutil  # High-level file operations, including copying and removing files.
import psutil  # Retrieves information on system utilization and resources.
from colorama import Fore  # Adds color to console output
from dask.distributed import (
    Client,
    LocalCluster,
)  # Dask's distributed computing framework for parallel processing.

# =========== Logger Setup =================
logger.remove()  # Remove the default logger configuration

logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
)  # Add a new logger configuration with custom formatting

logger.add(
    "logs/thesis_project_{time:YYYY-MM-DD_HH-mm-ss}.log", enqueue=True
)  # new log file for every run, named with a timestamp


# =========== Timer Decorator ===========
def timer_decorator(func):
    """
    Decorator which wraps a function to measure execution time or log performance details.

    Features
    ---------------------------------
        - Measures and logs function execution time when enabled.
        - Preserves the original function’s metadata (e.g., name, docstring) using `@wraps`.
        - Works seamlessly with any function, maintaining its arguments and return values.

    Parameters
    ---------------------------------
        func : callable
            The function being decorated.

    Returns
    ---------------------------------
        callable:
            A wrapped function that executes the original function while optionally
            recording its execution time.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        """
        Function which ensures that the wrapped function maintains its expected return type and behavior
        while optionally allowing execution time measurement for performance tracking.

        Parameters
        ---------------------------------
            - *args : tuple
                Positional arguments passed to the wrapped function.
            - **kwargs : dict
                Keyword arguments passed to the wrapped function.

        Returns
        ---------------------------------
            Any:
                The output of the wrapped function, ensuring normal usage without modification.
        """

        # start_time = time.time()  # Capture the start time before executing the function. (Optional)

        result = func(*args, **kwargs)  # Execute the original function.

        # end_time = time.time() # Capture the end time after executing the function. (Optional)
        # logger.debug(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds.") # Log the execution time of the function. (Optional)

        return result  # Return the original function’s result unchanged.

    return wrapper  # Return the wrapped function for execution.


# =========== Signal Handling ===========
@timer_decorator
def signal_handler(signum: int, frame) -> None:
    """
    Function which handles and processes signals received by the program. It ensures a graceful shutdown
    by cleaning up resources, terminating child processes, and logging the shutdown signal.

    Features
    ---------------------------------
        - Handles various signals (e.g., SIGINT, SIGTERM) to trigger cleanup operations.
        - Logs an informational message when a shutdown signal is received.
        - Exits the program gracefully with status code `0` after cleanup.

    Parameters
    ---------------------------------
        - signum : int
            The signal number received by the program.
        - frame : Any
            The current stack frame at the time the signal was received.

    Raises Exception
    ---------------------------------
        If an error occurs during the cleanup process, it is logged and the program exits with status code `1'
    """
    try:  # Attempt to handle the signal and clean up resources.
        cleanup_resources()  # Clean up resources such as child processes and semaphores.

        logger.info(
            f"\nReceived shutdown signal, cleaning up..."
        )  # Log the shutdown signal received.

        sys.exit(0)  # Exit the program with status code `0` after successful cleanup.

    except Exception as e:  # Handle any errors that occur during cleanup.
        logger.error(f"\nError during cleanup: {e}")  # Log the error during cleanup.
        sys.exit(
            1
        )  # Exit the program with status code `1` after an error during cleanup.


# =========== Resource Cleanup ===========
@timer_decorator
def cleanup_resources() -> None:
    """
    Function which cleans up resources such as child processes and semaphores to ensure a clean
    shutdown of the program. It terminates active child processes, removes orphaned semaphores,
    and triggers garbage collection to free up memory.

    Features
    ---------------------------------
        - Terminates all active child processes created by the current process.
        - Removes orphaned semaphores stored in `/dev/shm` (Unix-based systems).
        - Forces garbage collection to free up memory that is no longer in use.

    Raises Exception
    -------------------
        If an error occurs during cleanup, logs the issue and re-raises the exception.
    """
    try:  # Attempt to clean up resources and terminate child processes.
        for child in active_children():  # Iterate through all active child processes.
            try:  # Attempt to terminate each child process.

                child.terminate()  # Terminate the child process gracefully.

                child.join(
                    timeout=3
                )  # Wait for the process to terminate with a timeout of 3 seconds.

                if (
                    child.is_alive()
                ):  # Check if the process is still alive after termination.
                    child.kill()  # Forcefully kill the process if it did not terminate.

            except (
                Exception
            ) as e:  # Handle any errors that occur during process termination.
                logger.warning(
                    f"\nFailed to terminate process: {e}"
                )  # Log the termination failure.
                continue  # Continue with the loop even if termination fails.

        cleanup_auto_semaphores()  # Clean up orphaned semaphores and resource tracker processes.

        gc.collect()  # Trigger garbage collection to free up memory.

    except Exception as e:  # Handle any errors that occur during resource cleanup.
        logger.error(
            f"\nResource cleanup failed: {e}"
        )  # Log the resource cleanup failure.
        raise  # Re-raise the exception to indicate the cleanup failure.


# =========== Active Children Cleanup===========
@timer_decorator
def cleanup_auto_semaphores() -> None:
    """
    Function which automatically removes orphaned semaphores and terminates resource tracker processes.

    Features
    ---------------------------------
        - Cleans up orphaned semaphores stored in the `/dev/shm` directory.
        - Terminates resource tracker processes that may be running in the background.
        - Logs the cleanup status for each semaphore and process.

    Raises Exception
    ---------------------
        If an error occurs while removing semaphores or terminating processes,
        logs the issue but does not re-raise the exception.
    """
    try:
        if os.path.exists("/dev/shm"):  # Check if the `/dev/shm` directory exists.

            current_pid = os.getpid()  # Get the current process ID.

            sem_pattern = f"sem.{current_pid}"  # Define the semaphore pattern to match.

            cleaned = (
                0  # Initialize a counter to track the number of cleaned semaphores.
            )

            for item in os.listdir(
                "/dev/shm"
            ):  # Iterate through items in the `/dev/shm` directory.
                if (
                    sem_pattern in item
                ):  # Check if the item matches the semaphore pattern.
                    try:  # Attempt to remove the semaphore.
                        sem_path = os.path.join(
                            "/dev/shm", item
                        )  # Get the full path of the semaphore.

                        os.unlink(sem_path)  # Remove the semaphore file.

                        cleaned += 1  # Increment the cleaned semaphore counter.

                        logger.debug(
                            f"Cleaned semaphore: {item}"
                        )  # Log the cleaned semaphore at debug level.

                    except (
                        Exception
                    ) as e:  # Handle any errors that occur during semaphore cleanup.
                        logger.warning(
                            f"{Fore.YELLOW}\nFailed to clean semaphore {item}: {e}"
                        )  # Log the semaphore cleanup failure.
                        continue  # Continue with the loop even if cleanup fails.

            current_process = psutil.Process()  # Get the current process using psutil.

            for (
                child
            ) in (
                current_process.children()
            ):  # Iterate through child processes of the current process.
                try:  # Attempt to terminate resource tracker processes.
                    if (
                        "resource_tracker" in child.name().lower()
                    ):  # Check if the process is a resource tracker.
                        child.terminate()  # Terminate the resource tracker process.
                        logger.debug(
                            f"{Fore.GREEN}\nTerminated resource tracker process"
                        )  # Log the termination of the resource tracker process.

                except (
                    Exception
                ) as e:  # Handle any errors that occur during process termination.
                    logger.warning(
                        f"{Fore.YELLOW}\nFailed to terminate resource tracker: {e}"
                    )  # Log the resource tracker termination failure.
                    continue  # Continue with the loop even if termination fails.

    except Exception as e:  # Handle any errors that occur during semaphore cleanup.
        logger.warning(
            f"\nSemaphore cleanup failed: {e}"
        )  # Log the semaphore cleanup failure.
        pass  # Continue with the program execution even if semaphore cleanup fails.


# =========== Memory Limit Calculation ===========
@timer_decorator
def limit_memory() -> int:
    """
    Function which calculates the memory limit to be assigned to the Dask cluster based on the system's
    available memory. It computes the memory limit as a percentage of the available memory, ensuring
    that the cluster does not exceed the system's capacity.

    Features
    ---------------------------------
        - Retrieves the system's available memory using `psutil`.
        - Calculates the memory limit as a percentage of the available memory.
        - Provides a fallback to 100% of available memory if the calculation fails.

    Returns
    ---------------------------------
        int:
            The computed memory limit in bytes to be used by the Dask cluster

    Raises Exception
    ---------------------------------
        If an error occurs during memory limit computation, logs a warning message and returns a fallback value.
    """
    try:  # Attempt to calculate the memory limit based on available memory.
        available_memory = (
            psutil.virtual_memory().available
        )  # Get the system's available memory.

        limit = int(
            available_memory * 1.5
        )  # Calculate the memory limit as 1.5 times the available memory.

        if limit is None or limit <= 0:  # Check if the calculated limit is invalid.
            limit = int(
                available_memory * 1
            )  # Use 100% of available memory as the fallback limit.

        return limit  # Return the computed memory limit.

    except (
        Exception
    ) as e:  # Handle any errors that occur during memory limit calculation.
        logger.warning(
            f"\nCould not set memory limits: {e}"
        )  # Log the memory limit calculation failure.
        available_memory = (
            psutil.virtual_memory().available
        )  # Get the system's available memory.
        limit = int(
            available_memory * 1
        )  # Use 100% of available memory as the fallback limit.
        return limit  # Return the fallback memory limit.


# =========== Create Dask Client ===========
@timer_decorator
def create_dask_client() -> Client:
    """
    Function which creates a Dask client connected to a local cluster with optimized configuration.

    Features
    ---------------------------------
        - Determines the number of physical and logical CPU cores.
        - Sets the number of workers to the maximum of 1 or the number of physical cores minus 1.
        - Sets the number of threads per worker to the maximum of 1 or half the logical core count divided by the number of workers.
        - Limits memory usage to the available system memory.
        - Logs the Dask configuration details.

    Returns
    ---------------------------------
        A Dask client connected to a local cluster with the specified configuration.

    Raises Exception
    ---------------------------------
        If any error occurs during cluster setup, logs the issue and raises an exception.
    """
    try:  # Attempt to create a Dask client for distributed computing.
        physical_cores = (
            psutil.cpu_count(logical=False) or 1
        )  # Get the number of physical CPU cores.

        logical_cores = (
            psutil.cpu_count(logical=True) or 1
        )  # Get the total number of logical CPU cores.

        n_workers = max(
            1, physical_cores - 1
        )  # Set the number of workers to the maximum of 1 or the number of physical cores minus 1.

        if (
            logical_cores > physical_cores
        ):  # If the logical core count is greater than the physical core count, set the number of threads per worker to the maximum of 1 or half the logical core count divided by the number of workers.
            threads_per_worker = max(
                1, logical_cores // (n_workers * 2)
            )  # This is done to prevent oversubscription of threads.
        else:
            threads_per_worker = max(
                1, physical_cores // n_workers
            )  # Otherwise, set the number of threads per worker to the maximum of 1 or the number of physical cores divided by the number of workers.

        memory_limit = (
            limit_memory()
        )  # Limit the memory usage to the available system memory.

        logger.info(
            f"{Fore.CYAN}\n  Dask Configuration "
            f"{Fore.WHITE}\n--------------------------------------"
            f"{Fore.MAGENTA}\n- Physical Cores: {Fore.GREEN}{physical_cores}{Fore.MAGENTA}"
            f"\n- Logical Cores: {Fore.GREEN}{logical_cores}{Fore.MAGENTA}"
            f"\n- Workers: {Fore.GREEN}{n_workers}{Fore.MAGENTA}"
            f"\n- Threads per Worker: {Fore.GREEN}{threads_per_worker}{Fore.MAGENTA}"
            f"\n- Memory Limit: {Fore.GREEN}{memory_limit / (1024**3):.2f} GB{Fore.RESET}"
        )  # Log the Dask configuration details.
        logger.debug(
            "Dask cluster lifetime='6h' and lifetime_restart=True"
        )  # Log the Dask cluster lifetime and restart settings.

        # Build the LocalCluster using the values we just computed
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=str(memory_limit),
            lifetime="6h",
            lifetime_restart=True,
            dashboard_address=":8787",
        )  # Create a local Dask cluster with the specified parameters.

        return Client(cluster)  # Return a Dask client connected to the cluster.

    except Exception as e:  # If an error occurs during cluster setup
        logger.error(
            f"\nDask client creation failed: {str(e)}"
        )  # Log the error message.
        raise  # Raise an exception to halt execution and propagate the error.


# =========== Clean Project Outputs ===========
@timer_decorator
def clean_project_outputs(base_path: str) -> None:
    """
    Function which cleans up selected subdirectories inside the outputs folder
    without deleting the GROUND_TRUTH directory.

    Features
    ---------------------------------
        - Deletes files and subdirectories from the specified output folders.
        - Leaves the GROUND_TRUTH directory untouched.
        - Logs the cleanup status for each directory.

    Raises Exception
    ---------------------------------
        If an error occurs during the cleanup process, logs the issue and raises an exception.
    """
    try:  # Attempt to clean up the outputs directory.
        target_folders = [
            "exported_models",
            "exported_reputations",
            "outlier_data",
            "plots",
            "splitted_data",
        ]  # List of folders to clean up

        outputs_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "outputs")
        )  # Get the absolute path of the outputs directory

        for folder in target_folders:  # Iterate through each target folder
            full_path = os.path.join(
                outputs_path, folder
            )  # Construct the full path of the folder
            if os.path.exists(full_path):  # Check if the folder exists
                for item in os.listdir(
                    full_path
                ):  # Iterate through items in the folder
                    item_path = os.path.join(
                        full_path, item
                    )  # Construct the full path of the item
                    try:  # Attempt to remove the item
                        if os.path.isdir(item_path):  # Check if the item is a directory
                            shutil.rmtree(
                                item_path
                            )  # Remove the directory and its contents
                        else:
                            os.remove(item_path)  # Remove the file
                    except (
                        Exception
                    ) as e:  # Handle any errors that occur during item removal
                        logger.warning(
                            f"Failed to remove item in {folder}: {e}"
                        )  # Log the removal failure
                logger.info(
                    f"Cleaned output folder: {folder}"
                )  # Log the cleanup status
            else:
                logger.warning(
                    f"Output folder not found: {folder}"
                )  # Log if the folder does not exist

    except Exception as e:  # Handle any errors that occur during the cleanup process
        logger.error(
            f"Failed to clean outputs directory: {e}"
        )  # Log the cleanup failure
        raise  # Raise an exception to halt execution and propagate the error.

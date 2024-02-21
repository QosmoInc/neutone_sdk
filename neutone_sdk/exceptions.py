import logging
import os

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class NeutoneException(Exception):
    """
    Custom exception class for Neutone. This is used to wrap other exceptions with more
    information and tips when other, more cryptic exceptions are raised.
    """
    def __init__(self, message: str, trigger_type: type[Exception], trigger_str: str):
        """
        Args:
            message: The message to display when this exception is raised.
            trigger_type: The type of exception that triggers this exception.
            trigger_str: Text that must be in the message of the trigger exception.
        """
        super().__init__(message)
        self.trigger_type = trigger_type
        self.trigger_str = trigger_str

    def raise_if_triggered(self, orig_exception: Exception) -> None:
        """
        Raises this exception from the original exception (still includes the stack
        trace and information of the original exception) if it is of the trigger type
        and contains the trigger string in its message. Otherwise, raises the original
        exception.
        """
        if (isinstance(orig_exception, self.trigger_type)
                and self.trigger_str in str(orig_exception)):
            raise self from orig_exception
        else:
            raise orig_exception


# TODO(cm): constant for now, but if we need more of these we could use a factory method
INFERENCE_MODE_EXCEPTION = NeutoneException(
    message="""
    Your model does not support inference mode. Ensure you are not calling forward on
    your model before wrapping it or saving it using `save_neutone_model()`. Also, try 
    to make sure that you are not creating new tensors in the forward call of your
    model, instead pre-allocate them in the constructor. If these suggestions fail, try 
    creating and saving your model entirely inside of a `with torch.inference_mode():` 
    block.
    """,
    trigger_type=RuntimeError,
    trigger_str="Inference tensors cannot be saved for backward."
)

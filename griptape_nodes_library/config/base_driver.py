from abc import abstractmethod
from collections.abc import Callable
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes


class BaseDriver(DataNode):
    """Base class for driver nodes that need to manage parameters and validate configuration.

    This class provides a foundation for driver nodes by offering utility methods
    for managing parameters, updating traits, and validating API keys.
    """

    # -----------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------

    def __init__(self, **kwargs) -> None:
        """Initialize a BaseDriver instance.

        Args:
            kwargs (Any): Additional keyword arguments to initialize the base DataNode class.

        Example:
            driver = BaseDriver(name="ExampleDriver")
        """
        super().__init__(**kwargs)

        # Set by `_install_model_access` on subclasses whose model parameter is a
        # license-filtered dropdown; stays None on the ones offering free text or a
        # dynamically fetched list.
        self._model_access: ModelAccessComponent | None = None

        self.add_parameter(
            Parameter(
                name="driver",
                output_type="Any",
                default_value=None,
                tooltip="",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "model config"},
                # This will be a complex object that cannot serialize and could contain private keys; it needs to be assigned at runtime.
                serializable=False,
            )
        )

    # -----------------------------------------------------------------------------
    # Abstract Methods
    # -----------------------------------------------------------------------------
    # @abstractmethod TODO: https://github.com/griptape-ai/griptape-nodes/issues/872
    def _get_common_driver_args(self, params: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        """Gets common driver arguments from parameters.

        Subclasses should implement this method to extract and process
        arguments that are needed to instantiate their specific driver.

        Args:
            self: The instance of the driver class.
            params: Dictionary of parameter values.

        Returns:
            A dictionary of arguments to be passed to the driver constructor.
        """
        return {}

    @abstractmethod
    def process(self) -> None:
        """Abstract method to process the node's data and set output values.

        Subclasses must override this method to define how the driver is created
        and how outputs are populated.

        Example:
            def process(self):
                driver = MyRealDriver()
                self.parameter_output_values["driver"] = driver
        """
        msg = "Subclasses must implement the 'process' method."
        raise NotImplementedError(msg)

    # -----------------------------------------------------------------------------
    # Public API Methods
    # -----------------------------------------------------------------------------

    def params_to_sparse_dict(
        self,
        params: dict,
        kwargs: dict,
        param_name: str,
        target_name: str | None = None,
        transform: Callable | None = None,
    ) -> dict:
        """Add a parameter to kwargs if it exists in params, with optional transformation.

        Args:
            params (dict): Dictionary containing parameters
            kwargs (dict): Dictionary to add parameter to if it exists
            param_name (str): Name of the parameter to look for in params
            target_name (str, optional): Name to use in kwargs. If None, uses param_name
            transform (callable, optional): Function to transform the value

        Returns:
            dict: The updated kwargs dictionary
        """
        value = params.get(param_name)
        if value is not None:
            transformed_value = transform(value) if transform else value
            kwargs[target_name or param_name] = transformed_value
        return kwargs

    # -----------------------------------------------------------------------------
    # Internal Helper Methods
    # -----------------------------------------------------------------------------

    def _install_model_access(
        self,
        *,
        model_choices: list[str],
        default_model: str,
        param: str = "model",
        deprecated_values: dict[str, str] | None = None,
    ) -> None:
        """Turn the named model parameter into a license-filtered dropdown.

        Stands in for `_update_option_choices` on driver nodes: the component owns
        the `Options` trait (so the parameter must not already carry one), adds an
        inline refresh button, marks the models the caller's license denies, and
        moves the stored value off a denied default. The declared default is still
        applied here, so the parameter reads the same as it did before adoption.

        Args:
            model_choices: The provider model ids the node offers, in dropdown order.
            default_model: The choice to select by default; must be in `model_choices`.
            param: Name of the parameter to decorate.
            deprecated_values: Legacy value -> canonical `model_choices` entry map,
                needed only when the parameter used to store something other than
                the provider's model id (an old display label, a catalog key).
        """
        if default_model not in model_choices:
            msg = f"Default model '{default_model}' is not one of the offered choices."
            raise ValueError(msg)
        parameter = self.get_parameter_by_name(param)
        if parameter is None:
            msg = f"Cannot install model access on '{type(self).__name__}': no '{param}' parameter."
            raise ValueError(msg)

        parameter.default_value = default_model
        self.set_parameter_value(param, default_model)
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=parameter,
            model_choices=model_choices,
            default_model=default_model,
            deprecated_values=deprecated_values,
        )

    def _get_selected_model_id(self) -> str:
        """The provider model id the model dropdown currently stores.

        The dropdown stores the provider's own id for the model, so a driver
        built from this node's parameters passes the stored value upstream as-is.
        Reading it through here keeps the parameter's name in one place.
        """
        if self._model_access is None:
            return ""
        return self._model_access.selected_value or ""

    def _raise_if_model_denied(self) -> None:
        """Fail closed rather than hand a downstream node a driver the license denies.

        Called at the top of `process` on nodes with a license-filtered dropdown;
        no-ops on the nodes that never installed one.
        """
        if self._model_access is not None:
            self._model_access.raise_if_selection_denied()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Keep the model dropdown's denial badge in step with the selection."""
        if self._model_access is not None:
            self._model_access.on_value_set(parameter, value)
        return super().after_value_set(parameter, value)

    def _display_api_key_message(self, service_name: str, api_key_env_var: str, api_key_url: str | None) -> None:
        """Checks if the API key exists in the node configuration, displays a message if not.

        This method checks if the API key for a specific service is present
        in the node's configuration. It returns True if the key exists and
        is not empty, otherwise returns False.

        Args:
            service_name: The name of the service in the node configuration.
            api_key_env_var: The name of the key variable within the service config.
            api_key_url: An optional URL for users to visit to obtain the key,
                         included in the error message if provided.

        Returns:
            bool: True if the API key exists and is not empty, False otherwise.
        """
        message_param = self.get_parameter_by_name("message")
        if message_param is not None:
            api_key = GriptapeNodes.SecretsManager().get_secret(api_key_env_var)
            msg = f"⚠️ This node requires an API key from {service_name}\nPlease visit {api_key_url} to obtain a valid key and update your settings."
            message_param.default_value = msg
            ui_options = message_param.ui_options
            if not api_key:
                ui_options["hide"] = False
            else:
                ui_options["hide"] = True
            message_param.ui_options = ui_options

    def _validate_api_key(
        self, service_name: str, api_key_env_var: str, api_key_url: str | None
    ) -> list[Exception] | None:
        """Validates the presence and non-emptiness of a specific API key in config.

        Checks the node's configuration for the given key within the specified
        service. Returns a list containing an error if the key is missing or empty.

        Args:
            service_name: The name of the service in the node configuration.
            api_key_env_var: The name of the key variable within the service config.
            api_key_url: An optional URL for users to visit to obtain the key,
                     included in the error message if provided.

        Returns:
            A list of exceptions (KeyError or ValueError) if validation fails,
            otherwise None.
        """
        exceptions = []

        api_key = GriptapeNodes.SecretsManager().get_secret(api_key_env_var)
        if not api_key:
            msg = f"API Key ('{api_key_env_var}') for service '{service_name}' is missing."
            if api_key_url:
                msg += f" Please visit {api_key_url} to obtain a valid key and update your settings."
            else:
                msg += " Please provide a valid API key in your settings."
            exceptions.append(KeyError(msg))

        # Display a message to the user if the API key is missing or empty.
        self._display_api_key_message(
            service_name=service_name, api_key_env_var=api_key_env_var, api_key_url=api_key_url
        )

        return exceptions if exceptions else None

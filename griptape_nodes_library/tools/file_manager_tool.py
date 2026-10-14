from typing import Any, cast
from urllib.parse import urlsplit

import httpx
from griptape.utils.griptape_cloud import griptape_cloud_url
from griptape_nodes.drivers.cloud_credentials import BASE_URL_SETTING_NAME, DEFAULT_CLOUD_BASE_URL
from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

from griptape_nodes_library.tools.base_tool import BaseTool
from griptape_nodes_library.utils.cloud_credential_utils import (
    resolve_cloud_api_key,
)
from griptape_nodes_library.utils.griptape_cloud_headers import build_griptape_cloud_headers

LOCATIONS = ["Workspace Directory", "GriptapeCloud"]

API_KEY_ENV_VAR = "GT_CLOUD_API_KEY"
SERVICE = "Griptape"
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def buckets_url() -> str:
    """Return the bucket-list endpoint of the configured Griptape Cloud deployment.

    Raises:
        ValueError: If ``GT_CLOUD_BASE_URL`` is not ``https``, unless it points at a loopback host.
    """
    base_url = (
        GriptapeNodes.SecretsManager().get_secret(BASE_URL_SETTING_NAME, should_error_on_not_found=False)
        or DEFAULT_CLOUD_BASE_URL
    )
    parts = urlsplit(base_url)
    if parts.scheme != "https" and not (parts.scheme == "http" and parts.hostname in LOOPBACK_HOSTS):
        msg = f"Attempted to fetch buckets. Failed because {BASE_URL_SETTING_NAME} must be an https URL, got {base_url!r}."
        raise ValueError(msg)
    return griptape_cloud_url(base_url, "api/buckets")


class FileManager(BaseTool):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.workdir = GriptapeNodes.ConfigManager().get_config_value("workspace_directory")

        self.update_tool_info(
            value=f"""The FileManager tool can be given to an agent to help it perform file operations and uses your Workspace Directory by default.\n
({self.workdir}).""",
            title="FileManager Tool",
        )

        # TODO: (jason) Add back when GriptapeCloudFileManagerDriver is working https://github.com/griptape-ai/griptape-nodes/issues/1416
        self.add_parameter(
            Parameter(
                name="file_location",
                type="str",
                tooltip="The location of the files to be used by the tool.",
                default_value=LOCATIONS[0],
                traits={Options(choices=LOCATIONS)},
                ui_options={"hide": True},
            )
        )
        """
        self.add_parameter(
             Parameter(
                 name="bucket_id",
                 type="str",
                 tooltip="The location of the files to be used by the tool.",
                 default_value=self.bucket_list[0][0] if self.bucket_list else "",
                 traits={Options(choices=[name for name, _ in self.bucket_list])},
                 ui_options={"hide": True},
             )
         )
        self.swap_elements("tool", "bucket_id")
        """
        self.hide_parameter_by_name("off_prompt")

    def after_value_set(
        self,
        parameter: Parameter,
        value: Any,
    ) -> None:
        if parameter.name == "file_location":
            if value == LOCATIONS[1]:
                self.show_parameter_by_name("bucket_id")
            else:
                self.hide_parameter_by_name("bucket_id")

        return super().after_value_set(parameter, value)

    def get_bucket_list(self) -> list[tuple[str, str]]:
        """Get the list of buckets from Griptape Cloud API.

        Returns:
            list[tuple[str, str]]: List of tuples containing (bucket_name, bucket_id)
        """
        url = buckets_url()
        try:
            response = httpx.get(
                url,
                headers=build_griptape_cloud_headers(resolve_cloud_api_key(), attribution=False),
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            return [(bucket["name"], bucket["bucket_id"]) for bucket in data["buckets"]]
        except httpx.HTTPStatusError as e:
            msg = f"Failed to fetch buckets from Griptape Cloud: {e}"
            raise RuntimeError(msg) from e
        except Exception as e:
            msg = f"Error fetching buckets: {e}"
            raise RuntimeError(msg) from e

    def process(self) -> None:
        off_prompt = self.parameter_values.get("off_prompt", True)
        file_location = cast("str", self.parameter_values.get("file_location"))

        config: dict = {"tool_type": "FileManager", "off_prompt": off_prompt, "file_location": file_location}
        if file_location == LOCATIONS[1]:
            bucket_name = cast("str", self.parameter_values.get("bucket_id"))
            bucket_id = dict(self.get_bucket_list()).get(bucket_name)
            if not bucket_id:
                msg = f"Invalid bucket name: {bucket_name}"
                raise ValueError(msg)
            config["bucket_id"] = bucket_id

        self.parameter_output_values["tool"] = config

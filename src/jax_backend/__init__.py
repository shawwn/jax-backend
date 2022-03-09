from __future__ import annotations
# read version from installed package
from importlib.metadata import version
__version__ = version(__name__)
del version

from typing import Any, List, Dict, Sequence, Optional
import warnings
import os

from absl import logging
# Disable "WARNING: Logging before flag parsing goes to stderr." message
logging._warn_preinit_stderr = 0

# import tsp.compiler
# from tsp import runtime as tsp_runtime
from backport.dataclasses import dataclass, field, InitVar, KW_ONLY, MISSING
import dataclasses

from jax._src.lib import xla_client
from jax._src.lib import xla_bridge
import numpy as np

import jax
import jax.numpy as jnp
import abc

class CustomBackendBuffer(abc.ABC):
  def __init__(self, device: CustomBackendDevice):
    super().__init__()
    self._device = device

  def device(self) -> CustomBackendDevice:
    return self._device

  @abc.abstractmethod
  def to_py(self) -> np.ndarray:
    raise NotImplementedError

  def platform(self):
    return self.device().client.platform

  def copy_to_device(self, device: CustomBackendDevice) -> CustomBackendBuffer:
    raise NotImplementedError


class CustomBackendExecutable(abc.ABC):
  def __init__(self, client: CustomBackendClient):
    super().__init__()
    self.client = client

  def local_devices(self) -> List[CustomBackendDevice]:
    return self.client.local_devices()[0:1]

  @abc.abstractmethod
  def execute(self, arguments: Sequence[CustomBackendBuffer]) -> List[CustomBackendBuffer]:
    raise NotImplementedError


class CustomBackendDevice(abc.ABC):
  def __init__(self, client: CustomBackendClient, id: int):
    super().__init__()
    self.client = client
    self.id = id

  @property
  def process_index(self):
    return 0

class CustomBackendClient(abc.ABC):
  def __init__(self, devices: List[CustomBackendDevice], local_devices: List[CustomBackendDevice] = None):
    super().__init__()
    self._devices = list(devices)
    self._local_devices = local_devices if local_devices is not None else list(devices)

  def process_index(self):
    return 0

  @property
  def platform(self):
    return self.name()

  def device_count(self):
    return len(self.devices())

  def devices(self) -> List[CustomBackendDevice]:
    return list(self._devices)

  def local_devices(self) -> List[CustomBackendDevice]:
    return list(self._local_devices)

  @abc.abstractmethod
  def compile(self, computation: str, compile_options: xla_client.CompileOptions) -> CustomBackendExecutable:
    raise NotImplementedError


  @abc.abstractmethod
  def buffer_from_pyval(
      self,
      argument: Any,
      device: CustomBackendDevice,
      force_copy: bool = True,
      host_buffer_semantics: xla_client.HostBufferSemantics = xla_client.HostBufferSemantics.ZERO_COPY
  ) -> CustomBackendBuffer:
    # TODO(phawkins): TSP's python API will accept a numpy array directly but
    # may want to explicitly construct a lower level BufferView to avoid copies.
    # return TspBuffer(self, device, np.array(argument, copy=True))
    raise NotImplementedError


  @classmethod
  @abc.abstractmethod
  def name(cls):
    raise NotImplementedError

  @classmethod
  def factory(cls):
    return cls()

  @classmethod
  def register(cls):
    xla_bridge.register_backend_factory(cls.name(), cls.factory, priority=-100)


# def custom_backend_client_factory():
#   return CustomBackendClient()
#
# xla_bridge.register_backend_factory("custom", custom_backend_client_factory, priority=-100)
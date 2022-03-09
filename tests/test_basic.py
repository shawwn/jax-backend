from __future__ import annotations
import unittest

import os


os.environ['JAX_PLATFORM_NAME'] = 'mine'

from absl.testing import absltest
from jax._src import test_util as jtu

from jax._src.lib import xla_client
from jax._src.lib import xla_bridge

import jax
from jax import numpy as jnp
import numpy as np

from jax.config import config
config.parse_flags_with_absl()

import jax.interpreters.mlir as mlir
from jax._src.lib.mlir import ir

import jax_backend
from typing import List, Any, Sequence

class MyBackendBuffer(jax_backend.CustomBackendBuffer):
  def __init__(self, device: jax_backend.CustomBackendDevice, value):
    super().__init__(device)
    self.value = value

  def to_py(self) -> np.ndarray:
    return self.value

  def copy_to_device(self, device: jax_backend.CustomBackendDevice) -> jax_backend.CustomBackendBuffer:
    return MyBackendBuffer(device, self.value)

class MyBackendExecutable(jax_backend.CustomBackendExecutable):
  def execute(self, arguments: Sequence[MyBackendBuffer]) -> List[MyBackendBuffer]:
    return [arguments[0]]

class MyBackendDevice(jax_backend.CustomBackendDevice):
  pass

class MyBackendClient(jax_backend.CustomBackendClient):
  def __init__(self):
    devices = [MyBackendDevice(self, id=i) for i in range(8)]
    super().__init__(devices)

  @classmethod
  def name(cls):
    return "mine"

  def buffer_from_pyval(
      self,
      argument: Any,
      device: jax_backend.CustomBackendDevice,
      force_copy: bool = True,
      host_buffer_semantics: xla_client.HostBufferSemantics = xla_client.HostBufferSemantics.ZERO_COPY
  ) -> jax_backend.CustomBackendBuffer:
    return MyBackendBuffer(device, np.array(argument, copy=True))

  def compile(self, computation: str, compile_options: xla_client.CompileOptions) -> jax_backend.CustomBackendExecutable:
    return MyBackendExecutable(self)

MyBackendClient.register()

class TestCase(jtu.JaxTestCase):
  def test_basic(self):
    self.assertEqual(1, 1)
    self.assertEqual("mine", xla_bridge.get_backend().platform)
    self.assertEqual(8, len(jax.devices()))
    one = jax.device_put(1, jax.devices()[0])
    three = one + 2

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())

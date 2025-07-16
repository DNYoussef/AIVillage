import logging
import struct

import numpy as np

logger = logging.getLogger(__name__)

class GGUFReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.metadata = {}
        self.tensors = {}

    def read(self):
        logger.info(f"Reading GGUF file: {self.file_path}")
        with open(self.file_path, "rb") as f:
            # Read magic number and version
            magic = f.read(4)
            if magic != b"GGUF":
                logger.error(f"Invalid GGUF file: {self.file_path}")
                raise ValueError("Not a valid GGUF file")

            version = struct.unpack("<I", f.read(4))[0]
            logger.debug(f"GGUF version: {version}")

            # Read metadata
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            logger.debug(f"Number of tensors: {n_tensors}, Number of metadata key-value pairs: {n_kv}")

            for _ in range(n_kv):
                key_length = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_length).decode("utf-8")
                value_type = struct.unpack("<I", f.read(4))[0]
                value = self._read_value(f, value_type)
                self.metadata[key] = value

            # Read tensors
            for _ in range(n_tensors):
                name_length = struct.unpack("<Q", f.read(8))[0]
                name = f.read(name_length).decode("utf-8")

                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))

                dtype = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]

                self.tensors[name] = {
                    "dims": dims,
                    "dtype": dtype,
                    "offset": offset
                }
        logger.info(f"Successfully read GGUF file: {self.file_path}")

    def _read_value(self, f, value_type):
        if value_type == 0:  # uint8
            return struct.unpack("<B", f.read(1))[0]
        if value_type == 1:  # int8
            return struct.unpack("<b", f.read(1))[0]
        if value_type == 2:  # uint16
            return struct.unpack("<H", f.read(2))[0]
        if value_type == 3:  # int16
            return struct.unpack("<h", f.read(2))[0]
        if value_type == 4:  # uint32
            return struct.unpack("<I", f.read(4))[0]
        if value_type == 5:  # int32
            return struct.unpack("<i", f.read(4))[0]
        if value_type == 6:  # float32
            return struct.unpack("<f", f.read(4))[0]
        if value_type == 7:  # bool
            return bool(struct.unpack("<B", f.read(1))[0])
        if value_type == 8:  # string
            length = struct.unpack("<Q", f.read(8))[0]
            return f.read(length).decode("utf-8")
        logger.warning(f"Unsupported value type: {value_type}")
        raise ValueError(f"Unsupported value type: {value_type}")

    def get_tensor(self, name):
        if name not in self.tensors:
            logger.error(f"Tensor {name} not found")
            raise KeyError(f"Tensor {name} not found")

        tensor_info = self.tensors[name]
        with open(self.file_path, "rb") as f:
            f.seek(tensor_info["offset"])
            data = np.fromfile(f, dtype=self._get_numpy_dtype(tensor_info["dtype"]), count=np.prod(tensor_info["dims"]))
            return data.reshape(tensor_info["dims"])

    def _get_numpy_dtype(self, gguf_dtype):
        dtype_map = {
            0: np.uint8,
            1: np.int8,
            2: np.uint16,
            3: np.int16,
            4: np.uint32,
            5: np.int32,
            6: np.float32,
            7: np.bool_,
        }
        return dtype_map.get(gguf_dtype, np.float32)  # Default to float32 if unknown

class GGUFWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.metadata = {}
        self.tensors = {}

    def add_metadata(self, key, value):
        self.metadata[key] = value

    def add_tensor(self, name, tensor):
        self.tensors[name] = tensor

    def write(self):
        logger.info(f"Writing GGUF file: {self.file_path}")
        with open(self.file_path, "wb") as f:
            # Write magic number and version
            f.write(b"GGUF")
            f.write(struct.pack("<I", 1))  # Version 1

            # Write tensor and metadata counts
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.metadata)))

            # Write metadata
            for key, value in self.metadata.items():
                f.write(struct.pack("<Q", len(key)))
                f.write(key.encode("utf-8"))
                self._write_value(f, value)

            # Write tensor information
            offset = f.tell() + len(self.tensors) * (8 + 4 + 8 + 8)  # Estimate tensor data start
            for name, tensor in self.tensors.items():
                f.write(struct.pack("<Q", len(name)))
                f.write(name.encode("utf-8"))

                f.write(struct.pack("<I", len(tensor.shape)))
                f.write(struct.pack(f"<{len(tensor.shape)}Q", *tensor.shape))

                dtype = self._get_gguf_dtype(tensor.dtype)
                f.write(struct.pack("<I", dtype))
                f.write(struct.pack("<Q", offset))

                offset += tensor.nbytes

            # Write tensor data
            for tensor in self.tensors.values():
                tensor.tofile(f)
        logger.info(f"Successfully wrote GGUF file: {self.file_path}")

    def _write_value(self, f, value):
        if isinstance(value, (int, np.integer)):
            f.write(struct.pack("<I", 5))  # int32 type
            f.write(struct.pack("<i", value))
        elif isinstance(value, (float, np.floating)):
            f.write(struct.pack("<I", 6))  # float32 type
            f.write(struct.pack("<f", value))
        elif isinstance(value, bool):
            f.write(struct.pack("<I", 7))  # bool type
            f.write(struct.pack("<B", int(value)))
        elif isinstance(value, str):
            f.write(struct.pack("<I", 8))  # string type
            encoded = value.encode("utf-8")
            f.write(struct.pack("<Q", len(encoded)))
            f.write(encoded)
        else:
            logger.error(f"Unsupported value type: {type(value)}")
            raise ValueError(f"Unsupported value type: {type(value)}")

    def _get_gguf_dtype(self, numpy_dtype):
        dtype_map = {
            np.uint8: 0,
            np.int8: 1,
            np.uint16: 2,
            np.int16: 3,
            np.uint32: 4,
            np.int32: 5,
            np.float32: 6,
            np.bool_: 7,
        }
        return dtype_map.get(numpy_dtype, 6)  # Default to float32 if unknown

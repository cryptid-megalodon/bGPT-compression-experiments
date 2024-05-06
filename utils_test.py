import unittest

import utils


class TestCompressionDecompression(unittest.TestCase):
    def test_compress_decompress_cycle(self):
        # Test data
        test_data = (
            b"Hello world! " * 1000
        )  # Create a reasonably large and repetitive data block

        # Compress the data
        compressed_data = utils.compress_file_data(test_data)

        # Decompress the data
        decompressed_data = utils.decompress_file_data(compressed_data)

        # Verify that decompressed data matches the original data
        self.assertEqual(
            decompressed_data,
            test_data,
            "Decompressed data should be identical to the original data",
        )

    def test_compression_effectiveness(self):
        # Test data
        test_data = b"Hello world! " * 1000  # Repetitive data should compress well

        # Compress the data
        compressed_data = utils.compress_file_data(test_data)

        # Check that the compressed data is smaller than the original
        self.assertLess(
            len(compressed_data),
            len(test_data),
            "Compressed data should be smaller than original data",
        )

    def test_empty_data(self):
        # Test data
        empty_data = b""

        # Compress the empty data
        compressed_data = utils.compress_file_data(empty_data)

        # Decompress the compressed empty data
        decompressed_data = utils.decompress_file_data(compressed_data)

        # Check that the decompressed data is still empty and matches the original
        self.assertEqual(
            decompressed_data,
            empty_data,
            "Decompressed empty data should be identical to the original empty data",
        )


if __name__ == "__main__":
    unittest.main()

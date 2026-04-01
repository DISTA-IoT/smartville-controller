# This file is part of the "Smartville" project.
# Copyright (c) 2024 University of Insubria
# Licensed under the Apache License 2.0.
# SPDX-License-Identifier: Apache-2.0
# For the full text of the license, visit:
# https://www.apache.org/licenses/LICENSE-2.0

# Smartville is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License 2.0 for more details.

# You should have received a copy of the Apache License 2.0
# along with Smartville. If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

# Additional licensing information for third-party dependencies
# used in this file can be found in the accompanying `NOTICE` file.

import torch

class DynamicLabelEncoder:
    """
    Thread-safe dynamic label encoder.

    Warning:
        All access to this class's data should be done through its methods.
        Direct dictionary access is not thread-safe.
    """

    def __init__(self):
        self._label_to_int = {}
        self._int_to_label = {}
        self._current_code = 0


    def fit(self, labels):
        """
        returns the number of new classes found!
        """

        # get the new labels found in the batch
        # batch_labels  - changed labels - current labels
        new_labels = set(labels) - set(self._label_to_int.keys())

        for label in new_labels:
            self.add_class(label)

        return new_labels


    def add_class(self, label):

        if label in self._label_to_int:
            return

        self._label_to_int[label] = self._current_code
        self._int_to_label[self._current_code] = label
        self._current_code += 1


    def transform(self, labels):

        encoded_labels = []

        for label in labels:
            encoded_labels.append(self._label_to_int[label])

        return torch.tensor(encoded_labels)


    def inverse_transform(self, encoded_labels):
        decoded_labels = [self._int_to_label[code.item()] for code in encoded_labels]
        return decoded_labels


    def inverse_transform_to_str(self, encoded_labels):
        """
        Decode a tensor/array/list of encoded labels into string labels.
        """
        flat_labels = encoded_labels.reshape(-1).tolist() if hasattr(encoded_labels, "reshape") else encoded_labels
        return [str(self._int_to_label[int(code)]) for code in flat_labels]


    def get_codes_for_labels(self, labels):
        """Return encoded integer IDs for known natural-language labels."""
        return [self._label_to_int[label] for label in labels if label in self._label_to_int]


    def get_mapping(self):
        return self._label_to_int


    def get_labels(self):
        return list(self._label_to_int.keys())


    def update_label(self, new_label, logger):

        # the caller needs to know if he should add a replay buffer
        add_replay_buffer_signal = False

        if not new_label in self._label_to_int:

            logger.info(f'Proactively added {new_label}')
            self.add_class(new_label)
            add_replay_buffer_signal = True

        return add_replay_buffer_signal

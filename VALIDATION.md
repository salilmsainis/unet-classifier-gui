# Validation

Validation applies to the U-Net training and inference application in this
repository.

Automated coverage includes:

- numeric image/mask matching and dataset rejection cases;
- palette TIFF and LZW-compressed scientific image input;
- rejection of negative and out-of-range class IDs;
- CPU training, validation metrics, and best-checkpoint creation;
- safe checkpoint metadata loading and full-resolution tiled inference;
- probability normalization and exact output dimensions;
- image-level split separation and source-file immutability;
- responsive GUI worker success, failure, and cancellation paths;
- the Train classifier and Apply classifier tabs, including single-image and
  directory-batch inference; and
- optional forward/backward checks on available MPS and CUDA processors.

Latest local validation: 2026-11-05 on Python 3.12.14 and PyTorch 2.14.0. All eight
applicable tests passed; MPS and CUDA tests were skipped because those processors were
not available in the isolated test environment.

The test suite is a software-mechanics check. It does not establish that a model is
accurate enough for a particular scientific application. Validate trained models on
independent, representative specimens before use.

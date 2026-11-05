# Security policy

## Supported version

Security fixes are applied to the latest version on the default branch.

## Reporting a vulnerability

Please do not disclose a suspected vulnerability in a public issue. Use GitHub's
private vulnerability reporting or security-advisory feature for this repository.
Include the affected file format, reproduction steps, expected result, and observed
result. Remove confidential microscopy data and credentials from any sample files.

This application processes images, masks, and classifier checkpoints selected by the
user and does not use a network connection after installation. Treat files from
untrusted sources cautiously and keep the declared image-size limits enabled.
Classifier checkpoints are loaded with PyTorch's restricted `weights_only=True`
mode, but should still come from a source you trust.

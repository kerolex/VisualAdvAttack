# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## 2025-01-08

### Added
* Saving of original image resized and upsampled for comparison with adversarial image
* Documentation of configure_optmiser
* Raise exception in configure_optmizer

### Changed

### Removed
* Autograd Variable (deprecated)

### Fixed
* Output resolution
* Clamp inside for loop (making the adversarial image noticeable)
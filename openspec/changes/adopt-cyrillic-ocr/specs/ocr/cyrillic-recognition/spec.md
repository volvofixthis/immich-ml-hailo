## Purpose

Provide reliable Cyrillic and Russian text recognition in the OCR pipeline while preserving the existing Hailo-accelerated text detection behavior.

## ADDED Requirements

### Requirement: Recognize Cyrillic text

The OCR pipeline SHALL use a recognition model and character dictionary matched for Cyrillic/Slavic text, English, and digits.

#### Scenario: Russian text recognition

- **WHEN** an image contains clear Russian Cyrillic text
- **THEN** the OCR response returns the recognized Cyrillic text with a confidence score

#### Scenario: Mixed Cyrillic and Latin text

- **WHEN** an image contains Cyrillic, Latin, and numeric characters
- **THEN** the OCR response preserves the recognized characters in reading order

### Requirement: Match recognition assets

The recognition HEF and character dictionary SHALL be generated from and used with the same model vocabulary and output class ordering.

#### Scenario: Matching model and dictionary

- **WHEN** the Cyrillic recognition HEF and dictionary are installed
- **THEN** CTC output indices decode to the intended Cyrillic characters

#### Scenario: Mismatched assets

- **WHEN** the recognition HEF or dictionary does not match the expected vocabulary
- **THEN** deployment validation fails before the model is used for OCR

### Requirement: Preserve Hailo text detection

The OCR pipeline SHALL continue using the existing Hailo text detection stage and SHALL pass its detected text regions to the Cyrillic recognition stage.

#### Scenario: Text region recognition

- **WHEN** Hailo text detection finds one or more regions
- **THEN** each region is normalized and passed to Cyrillic recognition

#### Scenario: No text regions

- **WHEN** text detection finds no regions
- **THEN** OCR returns an empty result without invoking recognition

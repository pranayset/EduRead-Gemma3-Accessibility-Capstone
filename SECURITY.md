# Security Policy

## Supported project

This repository is an educational research and portfolio project. Do not use it to process confidential student records, personal data, or restricted educational materials.

## Reporting a vulnerability

Please report security issues privately to the repository maintainer through GitHub's private vulnerability reporting feature, if enabled. Do not publish access tokens, private datasets, model credentials, or exploitable details in a public issue.

## Credential and data handling

- Store Hugging Face and other access tokens in Colab Secrets or a local environment file.
- Never commit `.env`, credentials, service-account files, private datasets, model weights, adapters, or checkpoints.
- Review notebook outputs before committing; outputs can contain credentials, private data, or large embedded content.
- Use only synthetic, public, or appropriately licensed sample data in the public repository.

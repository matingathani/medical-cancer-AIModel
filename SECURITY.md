# Security

## Reporting a vulnerability

If you believe you have found a security vulnerability (e.g. unsafe deserialization, credential exposure, or dependency issues), please **do not** open a public issue.

- **Preferred:** Email the maintainers (see GitHub repo "About" or owner profile) with a description and steps to reproduce.
- We will acknowledge and work on a fix. Please allow reasonable time before any public disclosure.

## Scope

- This project is for **research and education** and is not a medical device. Security issues that could lead to **remote code execution**, **data exposure**, or **supply-chain** problems are in scope.
- **Model robustness / adversarial examples** are research topics; please open a normal issue if you want to discuss.

## Data

- Do not commit patient data, API keys, or secrets. The repo uses `.gitignore` for `data/`, `config.yaml`, and `.env`. If you accidentally commit sensitive data, contact maintainers immediately.

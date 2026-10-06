# Privacy statement — Uncertainty Lab v1.0.0

## Bundled plugin behavior

The bundled Uncertainty Lab source is self-contained. Static review of the public candidate found no HTTP client, socket client, subprocess execution path, embedded credential, API key, private key, or developer-operated telemetry endpoint in the executable Python source.

The plugin does not intentionally transmit user measurement data, models, samples, covariance matrices, posterior draws, or results to a developer-operated external service.

Some workflows can read user-supplied local files available to the execution environment, for example fit/covariance inputs. Those files are used for the requested calculation and provenance hashing.

## Platform handling

This statement describes the bundled Uncertainty Lab code. Storage, retention, account, and service-level handling performed by ChatGPT/OpenAI is governed by the applicable OpenAI product terms and privacy policies rather than by this package.

## External references

Documentation may contain links or citations to methodological sources such as BIPM/JCGM. The bundled Python engines do not fetch those links during ordinary calculations.

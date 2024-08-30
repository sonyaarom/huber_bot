# QA Generator and Evaluator

This folder contains a system for generating and evaluating question-answer pairs from text data. It uses a combination of local language models and OpenAI's API to create and assess the quality of questions based on given text input.

## Purpose

The main goals of this project are:

1. Generate relevant question-answer pairs from input text data.
2. Evaluate the quality of generated questions based on specificity, realism, and clarity.
3. Provide a flexible and easy-to-use interface for running the system with various configurations.

## Contents

- `qa_generator_evaluator.py`: The main Python script that handles generation and evaluation of QA pairs.
- `Makefile.qa`: A Makefile to simplify running the script with different options.

## Prerequisites

- Python 3.7+
- OpenAI API key
- Required Python packages (ensure these are installed in your environment)

## Setup

1. Ensure you have Python 3.7+ installed on your system.
2. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Using the Makefile

The `Makefile.qa` provides several targets to run the QA generator and evaluator with different options. Here are the main commands:

### Run without sampling

```
make -f Makefile.qa run
```

This will run the script using the default input file, output file, and maximum number of questions.

### Run with sampling

```
make -f Makefile.qa run-sample SAMPLE_SIZE=100
```

This will run the script with a sample of 100 rows from the input data.

### Customize input, output, and number of questions

```
make -f Makefile.qa run INPUT_FILE=custom_data.csv OUTPUT_FILE=results.csv MAX_QUESTIONS=10
```

This allows you to specify custom input and output files, as well as the maximum number of questions to generate.

### Clean up

```
make -f Makefile.qa clean
```

This will remove the generated output files.

### Get help

```
make -f Makefile.qa help
```

This will display information about available targets and variables.

## Customizing the Run

You can customize various aspects of the run by modifying the variables in the make command. For example:

```
make -f Makefile.qa run-sample SAMPLE_SIZE=50 MAX_QUESTIONS=20 INPUT_FILE=my_data.csv OUTPUT_FILE=my_results.csv
```

This will:
- Use `my_data.csv` as the input file
- Sample 50 rows from the input
- Generate up to 20 questions per sampled text
- Save the results in `my_results.csv`

## Note

Remember to always use `-f Makefile.qa` when running make commands, as the Makefile is not named with the default "Makefile" name.
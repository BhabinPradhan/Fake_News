# MOSAIC — Cross-Domain Multimodal Fake News Detection

> University of Windsor COMP 4990 4th Year Capstone Project

## What It Does
Our MOSAIC program takes a social media post (a text and image) and classifies it as either **Real**, **Fake**, or **Uncertain**.
It works by running the input through an ensemble of 28 expert models spreading across 7 different model families
and 4 dataset domains, which we then combine their votes using weighted ensemble logic. 
Rather than forcing a falsely confident guess on inputs that might seem ambiguous, the system returns `Uncertain` when vote strength or
agreement falls below a confidence threshold. This is an intentional feature that was added, not a failure.

Keep note that MOSAIC is a pattern-based detector trained on misinformation datasets. It is not a live
fact-checker and does not verify claims against external sources in real time.

## System Architecture
<!-- Amer: diagram or description of model_manager → api.py → index.html flow -->

## Project Structure
<!-- Amer: annotated file tree of core files -->

## Setup & Installation
<!-- Amer or shared: Python version, pip install, how to run api.py -->

## Running the Demo
<!-- Bhabin: ngrok command, local test path, what to expect -->

## API Endpoints
<!-- Marc: /health, /predict, /scrape with examples -->

## Benchmark Results
<!-- Darren: summary of 100-case aggregate, point to CSV files -->

## Known Limitations
<!-- Amer: paste/adapt from ProjectSummary.txt limitations section -->

## Team
<!-- Everyone: names and roles -->

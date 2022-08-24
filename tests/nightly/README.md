# Nightly Test

The nightly test is based on Docker, and currently running on AWS A100 instances. It will run the entire unittests and model benchmarks to track the test status and performance regression.

## Before Start

Before start, need to build the docker with following command,
```
AITemplate$ bash docker/build.sh cuda
```
This will build current code to a Docker image with tag `ait:latest`

You also need a personal [Github Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)

## Testing & Submitting Results inside Docker

This mode is usually used for debugging.

```
docker run --gpus all -it ait:latest bash
```
This will create an interactive docker instance. After enter the image, run
```
bash /AITemplate/tests/nightly/test_all.sh
```
After a few hours, when all tests are finished, run
```
NIGHTLY=1 python3 /AITemplate/test/nightly/report_cuda.py --github-token xxxxxx
```
It will create issue of unittests results and benchmark results.

## Testing in Docker, Submitting outside of Docker

This mode is usually used in automated tests. Run

```
docker run --gpus all ait:latest bash /AITemplate/tests/nightly/test_all.sh 1>log
```

After a few hours the tests and benchmark results will be inside `log` file

Then submit results with the command.

```
python3 AIT_PATH/test/nightly/report_cuda.py --code-path AIT_PATH --unittest-log log --benchmark-log log --github-token xxxxx
```

## Bisect in Docker

1. Build Debug Docker Image
```bash docker/build.sh cuda debug```


2. Launch an interactive docker instance
```docker run --gpus all -it ait:latest bash```

3. Run Bisect script
```
cd /AITemplate
python3 tests/nightly/bisect.py
```

4. Sample Input
```
Old Commit ID:
New Commit ID:
Test Command: bash /AITemplate/tests/nightly/benchmark.sh
```

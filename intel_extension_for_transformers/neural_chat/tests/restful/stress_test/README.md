Stress Test for Restful API of Neural Chat
====================

The [Locust](https://github.com/locustio/locust) framework is used for stress test.

## Preperations

### Modify Configurations

Before running stress test, you could modify `stress_test/locust.conf`:
- locustfile: The stress test script to run.
- headless: Do not use web UI, show test result in terminal directly.
- host: The host IP and port of the backend server.
- users: Number of users you want to mock.
- spawn-rate: Rate to spawn users.
- run-time: Stop after the specified amount of time, e.g. (300s,
                        20m, 3h, 1h30m, etc.).

For more configuration settings, refer to the [official documentations](https://docs.locust.io/en/stable/configuration.html) of locust.

### install locust package

```bash
pip install locust
```



## Run Stress Test
```
cd stress_test
locust
```
![stress test result](https://i.imgur.com/iCWkUQ6.jpeg)
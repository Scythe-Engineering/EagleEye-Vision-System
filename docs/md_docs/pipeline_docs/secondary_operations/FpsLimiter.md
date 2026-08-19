# FpsLimiter

`FpsLimiter` delays pipeline iterations to cap their rate, then passes the input through unchanged.

## Inputs

`data` is the value to pace.

## Outputs

`data` is the same value received by the operation.

## When to use

Use this when a source does not already provide suitable pacing.

## Configuration

- `fps`: Target rate. Default `30.0`; range `0.1` to `120.0`.

## Limitations

The operation sleeps on the pipeline thread. It caps the rate but cannot make a slow pipeline reach the target.

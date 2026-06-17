let lastError: unknown = undefined;

const originalConsoleError = console.error;
console.error = (...args: unknown[]) => {
  if (args[0] instanceof Error) {
    lastError = args[0];
  }
  originalConsoleError.apply(console, args);
};

export function consumeLastCapturedError(): Error | undefined {
  const err = lastError instanceof Error ? lastError : undefined;
  lastError = undefined;
  return err;
}

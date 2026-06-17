export function reportLovableError(
  error: Error,
  context?: Record<string, string>,
): void {
  console.error("[lovable]", error, context);
}

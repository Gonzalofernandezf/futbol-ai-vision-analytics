export function LoadingSkeleton() {
  return (
    <div className="flex flex-1 flex-col gap-4 p-6">
      <div className="h-20 animate-pulse rounded-lg bg-muted" />
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr_350px]">
        <div className="h-[460px] animate-pulse rounded-lg bg-muted" />
        <div className="h-[460px] animate-pulse rounded-lg bg-muted" />
      </div>
    </div>
  );
}

export function ErrorPanel({ message }: { message: string }) {
  return (
    <div className="flex flex-1 items-center justify-center p-6">
      <div className="max-w-sm rounded-lg border border-destructive/40 bg-destructive/10 p-6 text-center">
        <h2 className="text-lg font-semibold text-foreground">No se pudo cargar el partido</h2>
        <p className="mt-2 text-sm text-muted-foreground">{message}</p>
      </div>
    </div>
  );
}
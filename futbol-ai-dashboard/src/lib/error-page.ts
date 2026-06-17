export function renderErrorPage(): string {
  return `<!DOCTYPE html>
<html lang="es">
<head><meta charset="utf-8"><title>Error</title></head>
<body style="font-family:system-ui;display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
  <div style="text-align:center">
    <h1>500</h1>
    <p>Ocurrió un error en el servidor. Recarga la página.</p>
  </div>
</body>
</html>`;
}

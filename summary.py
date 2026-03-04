import os

# Configuración
OUTPUT_FILE = "proyecto_completo.txt"
# Carpetas y archivos a IGNORAR (para no llenar el chat de basura)
IGNORE_DIRS = {
    '__pycache__', '.git', '.vscode', 'venv', 'env', 
    'output_videos', 'runs', 'stubs', 'inference'
}
IGNORE_EXTENSIONS = {'.mp4', '.avi', '.pt', '.pkl', '.csv', '.jpg', '.png'}
IGNORE_FILES = {'pack_project.py', 'football_analysis_land_on_moon.mp4'}

def pack_project():
    root_dir = os.getcwd()
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        # 1. Escribir estructura del directorio primero
        f_out.write("ESTRUCTURA DEL PROYECTO:\n")
        f_out.write("========================\n")
        for root, dirs, files in os.walk(root_dir):
            # Filtrar carpetas ignoradas
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            level = root.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * (level)
            f_out.write(f'{indent}{os.path.basename(root)}/\n')
            subindent = ' ' * 4 * (level + 1)
            for file in files:
                if not any(file.endswith(ext) for ext in IGNORE_EXTENSIONS) and file not in IGNORE_FILES:
                    f_out.write(f'{subindent}{file}\n')
        
        f_out.write("\n\nCONTENIDO DE LOS ARCHIVOS:\n")
        f_out.write("==========================\n")

        # 2. Escribir contenido de cada archivo
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)
                
                # Solo procesamos archivos de código y config relevantes
                if ext in ['.py', '.html', '.css', '.js', '.json', '.txt', '.md']:
                    if file in IGNORE_FILES or "requirements" in file: continue
                    
                    # Ignorar json muy pesados si los hubiera
                    if ext == '.json' and os.path.getsize(file_path) > 1024 * 50: # Mayor a 50KB
                        f_out.write(f"\n--- ARCHIVO: {file} (Omitido por tamaño) ---\n")
                        continue

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f_in:
                            content = f_in.read()
                            f_out.write(f"\n\n{'='*50}\n")
                            f_out.write(f"ARCHIVO: {os.path.relpath(file_path, root_dir)}\n")
                            f_out.write(f"{'='*50}\n")
                            f_out.write(content)
                    except Exception as e:
                        print(f"No se pudo leer {file}: {e}")

    print(f"✅ ¡Listo! Todo tu código está en '{OUTPUT_FILE}'.")
    print("Copia el contenido de ese archivo y pégalo en tu nuevo chat.")

if __name__ == "__main__":
    pack_project()
import { useRef } from "react";
import { Camera } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Props {
  onFile: (file: File) => void;
  label?: string;
}

export function AvatarUploader({ onFile, label = "Subir avatar" }: Props) {
  const ref = useRef<HTMLInputElement>(null);
  return (
    <>
      <input
        ref={ref}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        aria-label={label}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
      <Button
        variant="ghost"
        size="icon"
        aria-label={label}
        title={label}
        onClick={() => ref.current?.click()}
      >
        <Camera className="h-4 w-4" />
      </Button>
    </>
  );
}

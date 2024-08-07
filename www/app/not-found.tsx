import { headers } from "next/headers";
import Link from "next/link";
import { permanentRedirect, usePathname } from "next/navigation";

export default function NotFound() {
  const heads = headers();

  const pathname = heads.get("next-url");

  if (pathname == null || pathname === "/") {
    permanentRedirect("/docs/core");
  }

  return (
    <div className="w-full h-screen flex flex-col items-center justify-center gap-4">
      <h2 className="text-lg">Not Found</h2>
      <p className="text-muted-foreground">Could not find requested resource</p>
      <Link href="/">Return Home</Link>
    </div>
  );
}

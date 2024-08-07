import { Logo } from "@/components/ui/icons";
import { BaseLayoutProps } from "fumadocs-ui/layout";
import { Building, GlobeIcon } from "lucide-react";

export const layoutOptions: BaseLayoutProps = {
  nav: {
    url: "/docs/core",
    title: (
      <>
        <Logo />
        <span className="text-foreground">DenserRetriever</span>
      </>
    ),
  },
  links: [
    {
      text: "DenserAI",
      url: "https://retriever.denser.ai",
      active: "nested-url",
      icon: <Building />,
    },
    {
      text: "About",
      url: "https://denser.ai/about",
      icon: <GlobeIcon />,
      external: true,
    },
  ],
};

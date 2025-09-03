"use client"
import {createContext, useContext, useState, useEffect} from "react";

const AppContext = createContext(null);

export const AppWrapper = ({
  children,
}: {
  children: React.ReactNode,
}) => {

  const [bootstrap, setBootstrap] = useState(undefined);

  useEffect(() => {

    if(!bootstrap) {

      const sync = async () => {

        const bootstrapJs = await import("bootstrap");
        setBootstrap(bootstrapJs);
      };
      sync();
    };
  }, [bootstrap]);

  return (

    <AppContext.Provider value={{bootstrap}}>

      {children}

    </AppContext.Provider>

  );
};

export const useAppContext = () => {
  
  return useContext(AppContext);
};
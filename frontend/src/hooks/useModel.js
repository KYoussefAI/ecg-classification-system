import { useEffect, useState } from "react";
import api from "../utils/api";

export default function useModel() {
  const [metadata, setMetadata] = useState(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    let active = true;
    const refresh = () =>
      api
        .get("/api/model")
        .then(({ data }) => {
          if (active) {
            setMetadata(data);
            setError(false);
          }
        })
        .catch(() => {
          if (active) {
            setMetadata(null);
            setError(true);
          }
        });
    refresh();
    const interval = setInterval(refresh, 30000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);
  return { metadata, error };
}

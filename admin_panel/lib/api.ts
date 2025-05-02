

export async function fetchData() {
  const response = await fetch("https://adaptfl-server.onrender.com/get_data");
  if (!response.ok) {
    throw new Error("Failed to fetch data");
  }
  return await response.json();
}
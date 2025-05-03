

export async function fetchData() {
  const response = await fetch("https://securefl.onrender.com/get_data");
  if (!response.ok) {
    throw new Error("Failed to fetch data");
  }
  return await response.json();
}
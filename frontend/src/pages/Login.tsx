import { useState } from "react";
import { login } from "../utils/apiUtils";
import { Link } from "raviger";

export default function Login() {
  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError("");
    try {
      const data = await login(username, password);
      localStorage.setItem("token", data.token);
      window.location.href = "/";
      // navigate("/")
    } catch (error) {
      setError("Invalid username/password.");
    }
  };

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [obsecureText, setObsecureText] = useState("password");

  return (
    <div className="inline-block ml-10 ">
      <div className="mx-auto">
        <div className="bg-[#272A30] mx-auto my-24 py-10 px-7 content-center rounded-xl">
          <div className="flex flex-col items-center text-gray-200 justify-center">
            <div>
              <form className="mx-5" onSubmit={handleSubmit}>
                <label htmlFor="username">Username</label>
                <input
                  autoFocus
                  type="text"
                  id="username"
                  name="username"
                  className=" focus:outline-none text-white  w-full bg-[#3b4046] rounded-lg p-2 my-2 flex-1"
                  value={username}
                  onChange={(_) => setUsername(_.target.value)}
                />

                <label className="inline-block " htmlFor="password">
                  Password
                </label>
                <div className=" pl-2 inline-block pt-2">
                  <button
                    type="button"
                    className="bg-[#3b4046] rounded-lg  p-1 text-center items-center "
                    onClick={() =>
                      obsecureText === "password"
                        ? setObsecureText("text")
                        : setObsecureText("password")
                    }
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      style={{ height: 20, width: 20 }}
                      fill="none"
                      viewBox="0 0 24 24"
                      strokeWidth={1.5}
                      stroke="currentColor"
                      className="w-6 h-6"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88"
                      />
                    </svg>
                  </button>
                </div>
                <input
                  type={obsecureText}
                  id="password"
                  name="password"
                  className=" focus:outline-none text-white  w-full bg-[#3b4046] rounded-lg p-2 my-2 flex-1"
                  value={password}
                  onChange={(_) => setPassword(_.target.value)}
                />
                <div className="text-center items-center">
                  {error && (
                    <p className="text-red-700 font-semibold text-md bg-red-200 w-full mt-3 rounded-lg px-3 py-2 text-center items-center">
                      {error}
                    </p>
                  )}
                </div>
                <div className="pt-2 font-semibold text-sm text-gray-300">
                  Want to register ?{" "}
                  <Link className="text-blue-500" href="/register">
                    Register Here
                  </Link>
                </div>
                <div className="flex flex-col">
                  <button
                    type="submit"
                    className="bg-blue-600 text-center text-white hover:bg-blue-700 shadow-blue-700/40 mt-10 shadow-lg  px-5 py-2 rounded-xl font-semibold"
                  >
                    Login
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

import { getProviders, signIn, ClientSafeProvider } from "next-auth/react";
import { GetServerSideProps } from "next";

interface SignInProps {
  providers: Record<string, ClientSafeProvider> | null;
}

export default function SignIn({ providers }: SignInProps) {
  return (
    <div>
      <h1>Sign In</h1>
      <div>
        {providers &&
          Object.values(providers).map((provider) => (
          <div key={provider.name}>
            <button onClick={() => signIn(provider.id)}>
              Sign in with {provider.name}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

export const getServerSideProps: GetServerSideProps = async (context) => {
  const providers = await getProviders();
  return {
    props: { providers },
  };
};
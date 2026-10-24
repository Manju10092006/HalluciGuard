import { proxyBackend } from '../../../../lib/backend-proxy.mjs';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 300;
async function handle(request, context) {
  const { path } = await context.params;
  return proxyBackend(request, path);
}
export { handle as GET, handle as POST };

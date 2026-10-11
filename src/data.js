import { loadDataServices } from './firebase'

// When a callable request never gets a readable response (function not
// deployed, emulator not running, network or CORS failure) the Functions SDK
// only reports a bare "internal" code, which means nothing to a user.
function friendlyError(error) {
  const code = String(error?.code || '').replace(/^functions\//, '')
  const message = String(error?.message || '').trim()
  const bare = !message || message.toLowerCase().startsWith(code)
  if (code === 'internal' && bare) {
    if (import.meta.env.DEV) {
      console.warn(
        'GreenGoals could not reach Cloud Functions. Deploy them with `firebase deploy --only functions`, or set VITE_USE_EMULATORS=true and run `firebase emulators:start`.',
      )
    }
    return new Error('We couldn’t reach GreenGoals right now. Check your connection and try again in a moment.')
  }
  if (code === 'deadline-exceeded' && bare) {
    return new Error('That took longer than expected. Please try again.')
  }
  if (code === 'unavailable' && bare) {
    return new Error('GreenGoals is briefly unavailable. Please try again in a moment.')
  }
  return new Error(message || 'Something went wrong. Please try again.')
}

async function callFunction(name, data = {}, options = {}) {
  const firebase = await loadDataServices()
  if (!firebase) throw new Error('Firebase is not configured.')
  try {
    const result = await firebase.functionsModule.httpsCallable(firebase.functions, name, options)(data)
    return result.data
  } catch (error) {
    console.error(`${name} failed`, error)
    throw friendlyError(error)
  }
}

export const initializeUser = () => callFunction('initializeUser')
export const generateChallenges = ({ forceRefresh = false } = {}) =>
  callFunction('generateChallenges', { forceRefresh }, { timeout: 310000 })
export const completeChallenge = (challengeId, proof = {}) =>
  callFunction('completeChallenge', {
    challengeId,
    proofText: proof.text || '',
    proofImage: proof.image || null,
  }, { timeout: 190000 })
export const createGroup = (name, type) => callFunction('createGroup', { name, type })
export const joinGroup = code => callFunction('joinGroup', { code })

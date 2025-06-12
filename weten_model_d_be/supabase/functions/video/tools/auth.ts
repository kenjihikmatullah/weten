import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.SUPABASE_URL || '';
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || '';
const supabase = createClient(supabaseUrl, supabaseKey);

export const validateAuthToken = async (req) => {
    const authHeader = req.headers.get('Authorization')!;
    const authToken = authHeader.replace('Bearer ', '');
    const { data: { user }, error } = await supabase.auth.getUser(authToken);
    if (error || !user) {
        throw new Error('Unauthorized');
    }
    return user;
};
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.39.0';

const supabaseUrl = Deno.env.get('SUPABASE_URL') || Deno.env.get('SB_URL');
const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') || Deno.env.get('SB_SERVICE_ROLE_KEY');
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
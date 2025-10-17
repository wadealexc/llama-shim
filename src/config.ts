import * as fs from 'fs';

type Config = {
    DEFAULT_MODEL: string,
    MODEL_FILES_PATH: string,
    LOG_DIR: string,
    SLEEP_AFTER_X_SECONDS: number,
};

export function readConfig(path: string): Config {
    const cfgJSON = JSON.parse(fs.readFileSync(path, 'utf-8'));

    return {
        DEFAULT_MODEL: cfgJSON['default'],
        MODEL_FILES_PATH: cfgJSON['models'],
        LOG_DIR: cfgJSON['logs'],
        SLEEP_AFTER_X_SECONDS: cfgJSON['sleepAfterXSeconds'],
    }
}
CREATE TABLE `config` (
	`id` integer PRIMARY KEY DEFAULT 1 NOT NULL,
	`name` text DEFAULT 'kitsu' NOT NULL,
	`enable_signup` integer DEFAULT true NOT NULL,
	`default_user_role` text DEFAULT 'user' NOT NULL,
	`jwt_expires_in` text DEFAULT '7d' NOT NULL,
	`updated_at` integer NOT NULL,
	CONSTRAINT "config_id_check" CHECK("config"."id" = 1)
);
--> statement-breakpoint
INSERT OR IGNORE INTO config (id, updated_at) VALUES (1, strftime('%s', 'now'));
